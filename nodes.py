import base64
import json
import os
import random
import re
from datetime import datetime
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
from moviepy.video.VideoClip import ImageClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips
from moviepy.video.fx.Resize import Resize
from moviepy.video.io.VideoFileClip import VideoFileClip
from config import Config
from state import AgentState


def _log_node_output(run_dir: str, node_name: str, payload: dict):
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "node_logs.jsonl")
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "node": node_name,
        "payload": payload
    }
    with open(log_path, "a", encoding="utf-8") as log_file:
        json.dump(entry, log_file, ensure_ascii=False)
        log_file.write("\n")


def _extract_title_from_content(raw_content: str, fallback: str) -> str:
    for line in raw_content.splitlines():
        candidate = line.strip().lstrip("#").strip()
        if candidate:
            return candidate[:120]
    return fallback


text_client = AzureOpenAI(
    api_key=Config.AZURE_TEXT_API_KEY,
    api_version=Config.AZURE_TEXT_API_VERSION,
    azure_endpoint=Config.AZURE_TEXT_ENDPOINT,
    azure_deployment=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
)

image_client = AzureOpenAI(
    api_key=Config.AZURE_IMAGE_API_KEY,
    api_version=Config.AZURE_IMAGE_API_VERSION,
    azure_endpoint=Config.AZURE_IMAGE_ENDPOINT,
    azure_deployment=Config.AZURE_IMAGE_DEVELOPMENT_NAME,
)

VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv")


def _list_movie_files() -> list[str]:
    movie_dir = Config.MOVIE_DIR
    if not movie_dir or not os.path.isdir(movie_dir):
        return []
    files = []
    for name in os.listdir(movie_dir):
        if name.lower().endswith(VIDEO_EXTENSIONS):
            files.append(os.path.join(movie_dir, name))
    return files


def fetch_articles_node(state: AgentState):
    """articleフォルダから記事を取得し、ナレーション用に要約・整形する"""
    target_articles = []
    start = datetime.strptime(state['start_date'], "%Y%m%d")
    end = datetime.strptime(state['end_date'], "%Y%m%d")
    run_dir = state.get('run_output_dir') or Config.OUTPUT_DIR

    if not os.path.exists(Config.ARTICLE_DIR):
        os.makedirs(Config.ARTICLE_DIR)

    # 1. ファイルのフィルタリング
    files_to_process = []
    for filename in os.listdir(Config.ARTICLE_DIR):
        match = re.match(r"(\d{8})_(.*)\.md", filename)
        if match:
            file_date_str, title = match.groups()
            file_date = datetime.strptime(file_date_str, "%Y%m%d")
            if start <= file_date <= end:
                files_to_process.append((filename, title, file_date_str))

    # 2. 各記事の読み込みと要約（ナレーション原稿作成）
    for filename, title, date_str in files_to_process:
        with open(os.path.join(Config.ARTICLE_DIR, filename), 'r', encoding='utf-8') as f:
            raw_content = f.read()

        # GPT-4oによる要約とナレーション整形
        # ここでURLの除去や自然な言い回しへの変換を指示
        response = text_client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT_NAME,  # GPT-4o用デプロイ名
            messages=[
                {"role": "system", "content": "あなたは優秀なニュースアナウンサーです。"},
                {"role": "user", "content": f"""
以下のニュース記事を、YouTubeショート用のナレーション原稿に要約してください。

【制約事項】
・500文字以内。
・URLや記号（URL、[ ]、( )など）は読み上げに適さないため、完全に削除または自然な言葉に置き換えること。
・視聴者が聞き取りやすいよう、専門用語は避け、自然な話し言葉（です・ます調）にすること。

記事タイトル: {title}
記事内容:
{raw_content}
"""}
            ]
        )

        summarized_content = response.choices[0].message.content.strip()
        human_title = _extract_title_from_content(
            raw_content, title.replace("_", " "))

        target_articles.append({
            'title': title,
            'display_title': human_title,
            'content': summarized_content,  # ここに綺麗な要約が入る
            'date': date_str
        })
        print(f"✅ 要約完了: {title}")

    _log_node_output(
        run_dir,
        "fetch_articles",
        {
            "article_count": len(target_articles),
            "article_titles": [article['display_title'] for article in target_articles]
        }
    )

    return {'articles': target_articles, 'run_output_dir': run_dir}


def generate_assets_node(state: AgentState):
    audio_paths = []
    image_paths = []
    image_prompts = []

    run_dir = state.get('run_output_dir') or Config.OUTPUT_DIR
    os.makedirs(run_dir, exist_ok=True)

    for i, article in enumerate(state['articles']):
        speech_config = speechsdk.SpeechConfig(
            subscription=Config.AZURE_SPEECH_KEY,
            region=Config.AZURE_SPEECH_REGION
        )
        speech_config.speech_synthesis_voice_name = "ja-JP-NanamiNeural"

        audio_filename = f"audio_{i}.wav"
        audio_path = os.path.join(run_dir, audio_filename)
        audio_config = speechsdk.audio.AudioOutputConfig(filename=audio_path)

        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=audio_config)
        synthesizer.speak_text_async(article['content']).get()
        audio_paths.append(audio_path)

        # 1. GPT-4oに「風景」としてのプロンプトを描写させる
        prompt_response = text_client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "あなたは映画のコンセプトアーティストです。ニュース記事を読み、"
                        "その背景を象徴するような『美しく広大な風景』の画像生成用プロンプトを英語で作成してください。"
                        "人物を直接描くのではなく、そのニュースが起きている場所や、"
                        "そのニュースがもたらす雰囲気を象徴する風景を描写してください。"
                    )
                },
                {
                    "role": "user",
                    "content": f"記事タイトル: {article.get('display_title') or article['title']}\n記事内容: {article['content']}"
                }
            ]
        )
        img_prompt = prompt_response.choices[0].message.content.strip()

        image_prompts.append(img_prompt)

        image_result = image_client.images.generate(
            model=Config.AZURE_IMAGE_DEVELOPMENT_NAME,
            prompt=f"{img_prompt} Digital art style, vibrant colors, 9:16 aspect ratio focus.",
            size="1792x1024",
            n=1,
            response_format="b64_json",
        )

        image_data = image_result.data[0]
        if image_data.url:
            image_paths.append(image_data.url)
        else:
            image_filename = f"image_{i}.png"
            image_path = os.path.join(run_dir, image_filename)
            with open(image_path, "wb") as img_file:
                img_file.write(base64.b64decode(image_data.b64_json))
            image_paths.append(image_path)

    _log_node_output(
        run_dir,
        "generate_assets",
        {
            "audio_files": audio_paths,
            "image_files": image_paths
        }
    )

    return {
        'audio_paths': audio_paths,
        'image_paths': image_paths,
        'image_prompts': image_prompts,
        'run_output_dir': run_dir
    }


def create_short_video_node(state: AgentState):
    clips = []
    run_dir = state.get('run_output_dir') or Config.OUTPUT_DIR
    os.makedirs(run_dir, exist_ok=True)
    output_path = os.path.join(run_dir, "final_youtube_short.mp4")
    movie_files = _list_movie_files()
    video_sources: list[VideoFileClip] = []
    article_visual_logs = []

    for i, article in enumerate(state['articles']):
        audio = AudioFileClip(state['audio_paths'][i])
        duration = audio.duration or 0
        duration = max(duration, 0.001)

        base_image = ImageClip(state['image_paths'][i], duration=duration)
        base_image = base_image.with_effects([Resize(height=1920)])

        def zoom_factor(t, total=duration):
            return 1 + 0.05 * (t / max(total, 0.001))

        base_image = base_image.with_effects([Resize(new_size=zoom_factor)])
        base_image = base_image.with_position("center")

        segments = []
        movie_segments_log = []

        image_intro_duration = min(5, duration)
        segments.append(base_image.with_duration(image_intro_duration))
        remaining = duration - image_intro_duration

        while remaining > 1e-3 and movie_files:
            movie_path = random.choice(movie_files)
            try:
                video_clip = VideoFileClip(movie_path)
                video_sources.append(video_clip)
            except Exception:
                continue

            clip_duration = min(5, remaining, video_clip.duration or 0)
            if clip_duration <= 0:
                video_clip.close()
                continue

            max_start = max(
                0, (video_clip.duration or clip_duration) - clip_duration)
            start = random.uniform(0, max_start) if max_start > 0 else 0

            segment = video_clip.subclipped(
                start_time=start,
                end_time=start + clip_duration
            ).with_audio(None)

            segment = segment.with_effects(
                [Resize(height=1920)]).with_position("center")
            segments.append(segment)

            movie_segments_log.append({
                "file": os.path.basename(movie_path),
                "start": round(start, 2),
                "duration": round(clip_duration, 2)
            })

            remaining -= clip_duration

        if remaining > 1e-3:
            segments.append(base_image.with_duration(remaining))

        article_video = concatenate_videoclips(
            segments, method="compose").with_duration(duration)
        article_video.audio = audio
        clips.append(article_video)

        article_visual_logs.append({
            "article": article.get('display_title') or article['title'],
            "movie_segments": movie_segments_log
        })

    final_video = concatenate_videoclips(clips, method="compose")

    try:
        final_video.write_videofile(
            output_path, fps=24, codec="libx264", audio_codec="aac"
        )
    finally:
        final_video.close()
        for source in video_sources:
            try:
                source.close()
            except Exception:
                pass

    _log_node_output(
        run_dir,
        "create_video",
        {
            "video_path": output_path,
            "clip_count": len(clips),
            "articles": article_visual_logs
        }
    )

    return {"video_path": output_path, 'run_output_dir': run_dir}
