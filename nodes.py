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
from moviepy.audio.AudioClip import concatenate_audioclips
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips
from moviepy.video.fx.Resize import Resize
from moviepy.video.io.VideoFileClip import VideoFileClip
from config import Config
from state import AgentState


def _log_node_output(run_dir: str, node_name: str, payload: dict):
    """Append a JSON line containing node metadata to the current run directory."""
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
    """Return the first non-empty line from raw markdown as a human friendly title."""
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
    """Collect usable background video files from the configured movie directory."""
    movie_dir = Config.MOVIE_DIR
    if not movie_dir or not os.path.isdir(movie_dir):
        return []
    files = []
    for name in os.listdir(movie_dir):
        if name.lower().endswith(VIDEO_EXTENSIONS):
            files.append(os.path.join(movie_dir, name))
    return files


def _format_date_label(date_str: str) -> str:
    """Convert YYYYMMDD into YYYY/MM/DD for display."""
    try:
        return datetime.strptime(date_str, "%Y%m%d").strftime("%Y/%m/%d")
    except ValueError:
        return date_str


def _format_date_range_label(start: str, end: str) -> str:
    """Create a human-friendly date range label."""
    start_label = _format_date_label(start)
    end_label = _format_date_label(end)
    return start_label if start == end else f"{start_label} - {end_label}"


def _clean_hashtag_text(text: str) -> str:
    """Normalize text so it can sit behind a YouTube hashtag."""
    cleaned = re.sub(r"[#＃]", "", text)
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = re.sub(r"[^\wぁ-んァ-ヶ一-龯ー]+", "", cleaned)
    return cleaned[:20].strip()


def _extract_hashtags(articles: list[dict]) -> list[str]:
    """Build a short list of hashtags derived from article titles."""
    hashtags: list[str] = []
    for article in articles:
        candidate = article.get('display_title') or article.get('title')
        if not candidate:
            continue
        cleaned = _clean_hashtag_text(candidate)
        if not cleaned or cleaned in hashtags:
            continue
        hashtags.append(cleaned)
        if len(hashtags) >= 5:
            break

    for fallback in ("ニュース", "ショート動画", "AI速報"):
        if len(hashtags) >= 5:
            break
        if fallback not in hashtags:
            hashtags.append(fallback)

    return hashtags[:5]


def _generate_youtube_metadata(state: AgentState) -> dict:
    """Compose a YouTube-ready title, description, and hashtags from the run state."""
    articles = state.get('articles', [])
    range_label = _format_date_range_label(
        state['start_date'], state['end_date'])

    if articles:
        title = f"{range_label}の主要ニュースTOP{len(articles)} | ショート解説"
        if len(articles) == 1:
            title = f"{range_label} {articles[0]['title'] } | ニュースショート"
    else:
        title = f"{range_label}のニュースダイジェスト"

    description_lines = [
        f"📅 収録期間: {range_label}",
        "",
        "📝 取り上げたトピック:"
    ]
    if articles:
        for article in articles:
            description_lines.append(
                f"- {article.get('display_title') or article['title']} ({_format_date_label(article.get('date', state['start_date']))})"
            )
    else:
        description_lines.append("- 該当する記事は見つかりませんでした。")

    description = "\n".join(description_lines).strip()
    hashtags = _extract_hashtags(articles)
    if articles:
        primary = articles[0]
        base_thumb = primary.get('display_title') or primary.get('title') or ""
    else:
        base_thumb = ""
    if base_thumb:
        thumbnail_title = base_thumb[:40]
    else:
        thumbnail_title = title[:40] if title else "最新ニュース"

    return {
        "title": title,
        "description": description,
        "hashtags": hashtags,
        "thumbnail_title": thumbnail_title
    }


def _format_timestamp(seconds: float) -> str:
    """Return SBV timestamp (H:MM:SS.mmm)."""
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3600 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours}:{minutes:02}:{secs:02}.{millis:03}"


def _split_sentences_for_captions(text: str) -> list[str]:
    """Split narration text into SBV-friendly chunks."""
    if not text:
        return []

    stripped = text.strip()
    if not stripped:
        return []

    # 「。」でのみ区切り、句読点を維持したまま抽出
    sentences = [
        chunk.strip()
        for chunk in re.findall(r'[^。]+。?', stripped)
        if chunk.strip()
    ]

    return sentences or [stripped]


def _build_sbv_caption(text: str, duration: float | None) -> str:
    """Generate SBV caption text with pseudo-timed segments."""
    sentences = _split_sentences_for_captions(text)
    if not sentences:
        sentences = ["（内容なし）"]

    total_chars = sum(len(s) for s in sentences) or 1
    total_duration = duration if duration and duration > 0 else len(
        sentences) * 3.0

    raw_durations = []
    for sentence in sentences:
        portion = max(0.8, (len(sentence) / total_chars) * total_duration)
        raw_durations.append(portion)

    scale = total_duration / \
        sum(raw_durations) if sum(raw_durations) > 0 else 1.0
    durations = [d * scale for d in raw_durations]

    lines = []
    cursor = 0.0
    for sentence, seg_duration in zip(sentences, durations):
        start_ts = _format_timestamp(cursor)
        end_ts = _format_timestamp(cursor + seg_duration)
        lines.append(f"{start_ts},{end_ts}")
        lines.append(sentence)
        lines.append("")
        cursor += seg_duration

    return "\n".join(lines).strip() + "\n"


def fetch_articles_node(state: AgentState):
    """Load dated markdown articles in range and summarize them for narration."""
    target_articles = []
    start = datetime.strptime(state['start_date'], "%Y%m%d")
    end = datetime.strptime(state['end_date'], "%Y%m%d")
    run_dir = state.get('run_output_dir') or Config.OUTPUT_DIR

    single_article_path = state.get("single_article_path")

    if not single_article_path and not os.path.exists(Config.ARTICLE_DIR):
        os.makedirs(Config.ARTICLE_DIR)

    # 1. ファイルのフィルタリング
    files_to_process: list[tuple[str, str, str]] = []
    if single_article_path:
        if not os.path.isfile(single_article_path):
            raise FileNotFoundError(f"記事ファイルが見つかりません: {single_article_path}")
        basename = os.path.basename(single_article_path)
        match = re.match(r"(\d{8})_(.*)\.md", basename)
        if match:
            file_date_str, title = match.groups()
        else:
            file_date_str = state['start_date']
            title = os.path.splitext(basename)[0]
        files_to_process.append((single_article_path, title, file_date_str))
    else:
        for filename in os.listdir(Config.ARTICLE_DIR):
            match = re.match(r"(\d{8})_(.*)\.md", filename)
            if match:
                file_date_str, title = match.groups()
                file_date = datetime.strptime(file_date_str, "%Y%m%d")
                if start <= file_date <= end:
                    files_to_process.append(
                        (os.path.join(Config.ARTICLE_DIR, filename), title, file_date_str))

    # 2. 各記事の読み込みと要約（ナレーション原稿作成）
    for filepath, title, date_str in files_to_process:
        with open(filepath, 'r', encoding='utf-8') as f:
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
・500文字以内
・冒頭で核心を伝える：「何が起きたのか」「なぜ重要なのか」を最初に明示
・その後、時系列や因果関係に沿って背景・経緯・影響を簡潔に説明
・URLや記号（[ ]、( )など）は完全に削除または自然な言葉に置き換える
・専門用語は使用可（ビジネスパーソン向け）だが、必要に応じて簡潔な補足を入れる
・です・ます調で統一し、遠回しな表現は避ける
・ビジネスへの影響や実務的な意味を優先的に含める

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
            "article_titles": [article['display_title'] for article in target_articles],
            "articles": target_articles
        }
    )

    return {'articles': target_articles, 'run_output_dir': run_dir}


def generate_audio_assets_node(state: AgentState):
    """Create narration audio files and scripts for each article."""
    audio_paths = []
    script_paths = []
    voice_outputs = []

    run_dir = state.get('run_output_dir') or Config.OUTPUT_DIR
    os.makedirs(run_dir, exist_ok=True)

    for i, article in enumerate(state['articles']):
        # 各記事ごとに Azure Speech を設定（音声スタイルを統一）
        speech_config = speechsdk.SpeechConfig(
            subscription=Config.AZURE_SPEECH_KEY,
            region=Config.AZURE_SPEECH_REGION
        )
        speech_config.speech_synthesis_voice_name = "ja-JP-NanamiNeural"

        audio_filename = f"audio_{i}.wav"
        audio_path = os.path.join(run_dir, audio_filename)
        audio_config = speechsdk.audio.AudioOutputConfig(filename=audio_path)

        # ナレーションを音声化し、ファイルへ保存
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=audio_config)
        synthesizer.speak_text_async(article['content']).get()
        audio_paths.append(audio_path)

        # 字幕のタイミング計算用に長さを取得（失敗しても無視）
        audio_duration = None
        try:
            temp_clip = AudioFileClip(audio_path)
            audio_duration = temp_clip.duration or None
        except Exception:
            audio_duration = None
        finally:
            try:
                temp_clip.close()
            except Exception:
                pass

        script_filename = f"script_{i}.sbv"
        script_path = os.path.join(run_dir, script_filename)
        # SBV 形式の字幕を生成し、YouTubeで直接使えるようにする
        captions_content = _build_sbv_caption(
            article['content'], audio_duration)
        with open(script_path, "w", encoding="utf-8") as script_file:
            script_file.write(captions_content)
        script_paths.append(script_path)

        voice_outputs.append({
            "index": i,
            "article_title": article.get('display_title') or article['title'],
            "audio_path": audio_path,
            "script_path": script_path,
            "spoken_text": article['content']
        })

    _log_node_output(
        run_dir,
        "generate_audio_assets",
        {
            "audio_files": [os.path.basename(p) for p in audio_paths],
            "script_files": [os.path.basename(p) for p in script_paths],
            "voice_outputs": [
                {
                    "index": entry["index"],
                    "article_title": entry["article_title"],
                    "spoken_text": entry["spoken_text"],
                    "audio_file": os.path.basename(entry["audio_path"]),
                    "script_file": os.path.basename(entry["script_path"])
                }
                for entry in voice_outputs
            ]
        }
    )

    return {
        'audio_paths': audio_paths,
        'script_paths': script_paths,
        'run_output_dir': run_dir
    }


def generate_visual_assets_node(state: AgentState):
    """Create illustrative prompts and images for each article."""
    image_paths = []
    image_prompts = []
    image_outputs = []

    run_dir = state.get('run_output_dir') or Config.OUTPUT_DIR
    os.makedirs(run_dir, exist_ok=True)

    for i, article in enumerate(state['articles']):
        # GPT に映画的な風景プロンプトを作らせる（FLUX用）
        prompt_response = text_client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "あなたは報道ビジュアルのコンセプトアーティストです。以下のニュース記事を読み、"
                        "その内容を視覚的に伝える画像生成用プロンプトを英語で作成してください。\n\n"

                        "【必須要素】\n"
                        "・ニュースの核心的な「モノ」「場所」「状況」を具体的に描写\n"
                        "・技術系ニュース：製品、デバイス、インフラ、デジタルインターフェース\n"
                        "・ビジネス系ニュース：オフィス空間、都市景観、企業ロゴのないビル群\n"
                        "・政策系ニュース：議場、公共施設、象徴的な建造物\n"
                        "・環境系ニュース：自然環境、気候現象、エコシステム\n\n"

                        "【禁止事項】\n"
                        "・実在の人物の顔や体（後ろ姿や遠景のシルエットは可）\n"
                        "・実在企業のロゴや商標\n"
                        "・特定可能な個人が写り込む構図\n\n"

                        "【推奨表現】\n"
                        "・抽象的なビジュアルメタファー（例：AIニュース→脳のニューラルネットワーク模様）\n"
                        "・象徴的なオブジェクト（例：半導体ニュース→マイクロチップのクローズアップ）\n"
                        "・環境や空間で状況を表現（例：経済危機→無人のオフィスフロア）\n\n"

                        "【スタイル指定】\n"
                        "・9:16縦型構図を意識\n"
                        "・ビジュアルジャーナリズム風の写実的スタイル\n"
                        "・色彩は記事のトーン（希望的/警告的/中立的）に合わせる\n"
                        "・視認性の高い明瞭な構図"
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

        # 9:16 比率のイラストを描画し、動画冒頭の静止画に使用
        image_result = image_client.images.generate(
            model=Config.AZURE_IMAGE_DEVELOPMENT_NAME,
            prompt=f"{img_prompt} Digital art style, vibrant colors, 9:16 aspect ratio focus.",
            size="1792x1024",
            n=1,
            response_format="b64_json",
        )

        image_data = image_result.data[0]
        image_location = None
        if image_data.url:
            image_location = image_data.url
            image_paths.append(image_location)
        else:
            image_filename = f"image_{i}.png"
            image_path = os.path.join(run_dir, image_filename)
            with open(image_path, "wb") as img_file:
                img_file.write(base64.b64decode(image_data.b64_json))
            image_location = image_path
            image_paths.append(image_location)

        image_outputs.append({
            "index": i,
            "article_title": article.get('display_title') or article['title'],
            "prompt": img_prompt,
            "image_path": image_location
        })

    _log_node_output(
        run_dir,
        "generate_visual_assets",
        {
            "image_files": [os.path.basename(p) if p else None for p in image_paths],
            "image_prompts": image_prompts,
            "image_outputs": [
                {
                    "index": entry["index"],
                    "article_title": entry["article_title"],
                    "prompt": entry["prompt"],
                    "image_file": os.path.basename(entry["image_path"]) if entry["image_path"] else None
                }
                for entry in image_outputs
            ]
        }
    )

    return {
        'image_paths': image_paths,
        'run_output_dir': run_dir
    }


def create_short_video_node(state: AgentState):
    """Combine generated images, stock footage, and narration into one short video."""
    clips = []
    run_dir = state.get('run_output_dir') or Config.OUTPUT_DIR
    os.makedirs(run_dir, exist_ok=True)
    output_path = os.path.join(run_dir, "final_youtube_short.mp4")
    movie_files = _list_movie_files()
    video_sources: list[VideoFileClip] = []
    article_visual_logs = []
    article_audio_clips: list[AudioFileClip] = []
    audio_timeline = []
    audio_cursor = 0.0

    for i, article in enumerate(state['articles']):
        audio = AudioFileClip(state['audio_paths'][i])
        article_audio_clips.append(audio)
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

        audio_timeline.append({
            "article": article.get('display_title') or article['title'],
            "audio_file": os.path.basename(state['audio_paths'][i]),
            "start": round(audio_cursor, 2),
            "duration": round(duration, 2)
        })
        audio_cursor += duration

    final_video = concatenate_videoclips(clips, method="compose")
    final_audio = None
    if article_audio_clips:
        final_audio = concatenate_audioclips(article_audio_clips)
        final_video.audio = final_audio

    try:
        final_video.write_videofile(
            output_path, fps=24, codec="libx264", audio_codec="aac"
        )
    finally:
        final_video.close()
        if final_audio is not None:
            try:
                final_audio.close()
            except Exception:
                pass
        for audio_clip in article_audio_clips:
            try:
                audio_clip.close()
            except Exception:
                pass
        for source in video_sources:
            try:
                source.close()
            except Exception:
                pass

    _log_node_output(
        run_dir,
        "create_video",
        {
            "video_file": os.path.basename(output_path),
            "clip_count": len(clips),
            "articles": article_visual_logs,
            "audio_timeline": audio_timeline
        }
    )

    return {
        "video_path": output_path,
        'run_output_dir': run_dir
    }


def generate_youtube_metadata_node(state: AgentState):
    """Produce a YouTube-ready title, description, and hashtag set."""
    run_dir = state.get('run_output_dir') or Config.OUTPUT_DIR
    os.makedirs(run_dir, exist_ok=True)

    metadata = _generate_youtube_metadata(state)
    hashtags_line = " ".join(f"#{tag}" for tag in metadata['hashtags']).strip()
    metadata_path = os.path.join(run_dir, "youtube_meta.txt")

    articles = state.get("articles") or []
    if articles:
        primary_title = articles[0].get(
            'display_title') or articles[0].get('title') or "最新ニュース"
    else:
        primary_title = "最新ニュース"
    note_line = f"- note: {primary_title}"
    zenn_line = f"- zenn: {primary_title}"

    spoken_block = "\n\n".join(
        article.get('content', '').strip()
        for article in articles if article.get('content')
    ).strip()

    sections = [
        "【内容】",
        note_line,
        zenn_line,
        "",
    ]

    if articles:
        for idx, article in enumerate(articles, start=1):
            title = article.get('display_title') or article.get(
                'title') or f"記事{idx}"
            spoken = article.get('content', '').strip()
            sections.append(f"{idx}. {title}")
            if spoken:
                sections.append(spoken)
            sections.append("")
    elif spoken_block:
        sections.append(spoken_block)
        sections.append("")

    sections.extend([
        "IT系の情報を発信しています。",
        "note",
        "https://note.com/kenquichi",
        "zenn",
        "https://zenn.dev/kenquichi",
        "",
        "エンジニアになる講座",
        "https://note.com/kenquichi/m/mc4926a77c1da",
        "",

        "\n".join(f"#{tag}" for tag in metadata['hashtags']) or "#ニュース"
    ])

    final_text = "\n".join(sections).rstrip() + "\n"

    with open(metadata_path, "w", encoding="utf-8") as meta_file:
        meta_file.write(final_text)

    _log_node_output(
        run_dir,
        "generate_youtube_metadata",
        {
            "metadata_file": os.path.basename(metadata_path),
            "title": metadata['title'],
            "description": metadata['description'],
            "hashtags": metadata['hashtags'],
            "hashtags_line": hashtags_line or "#ニュース #ショート動画",
            "thumbnail_file": os.path.basename(state.get('thumbnail_path')) if state.get('thumbnail_path') else None,
            "thumbnail_title": metadata['thumbnail_title']
        }
    )

    return {
        "youtube_metadata_path": metadata_path,
        "thumbnail_title": metadata['thumbnail_title'],
        'run_output_dir': run_dir
    }
