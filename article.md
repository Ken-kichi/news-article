

なぜ作ったか？
- ニュース解説記事を音声にして家事をしながら聞きたかった
- 毎日投稿しているニュース解説記事を動画にしたら動画でニュースを見る層にアプローチができる
- LangGraphで文章生成以外の用途に挑戦してみたかった

学習できるところ
LangGraphの学習
動画生成
音声生成
動画編集
Youtubeアップ
Canvaの使い方

## セットアップ
### プロジェクトフォルダの作成
```bash
mkdir news-article
cd news-article
uv init
uv venv
uv add azure-cognitiveservices-speech dotenv langchain langgraph openai langchain_openai moviepy typer
mkdir article movie output
touch config.py state.py nodes.py graph.py
```

movieは以下のサイトから取得します。


## Azureのセットアップ
### Azure AI Foundry
eastusで作成
FLUX.1-Kontext-pro　と　gpt-4.1　を作成
### Microsoft Foundry | 音声サービス
East USで作成

https://learn.microsoft.com/ja-jp/azure/ai-services/speech-service/get-started-text-to-speech?tabs=new-foundry%2Cmacos&pivots=programming-language-python

## config.pyの設定
### .envの作成
```txt
# テキスト生成
AZURE_TEXT_API_KEY=
AZURE_TEXT_ENDPOINT=

# 音声
AZURE_SPEECH_KEY=
AZURE_SPEECH_ENDPOINT=
AZURE_SPEECH_REGION=

# 画像生成
AZURE_IMAGE_KEY=
AZURE_IMAGE_ENDPOINT=

```
### config.pyの作成
```py:config.py
import os
from dotenv import load_dotenv

load_dotenv()


def _split_endpoint(endpoint: str | None) -> tuple[str | None, str | None]:
    """Return (resource_base_url, api_version) extracted from a raw endpoint."""
    if not endpoint:
        return None, None

    base = endpoint.strip()
    if not base:
        return None, None

    api_version = None
    if "api-version=" in base:
        api_version = base.split("api-version=")[-1].split("&")[0].strip()
    base = base.split("?")[0]
    if "/openai/" in base:
        base = base.split("/openai/")[0]

    normalized = base.rstrip("/") + "/"
    return normalized, api_version


_raw_text_endpoint = os.getenv("AZURE_TEXT_ENDPOINT")
_text_endpoint, _text_version = _split_endpoint(_raw_text_endpoint)

_raw_image_endpoint = os.getenv("AZURE_IMAGE_ENDPOINT")
_image_endpoint, _image_version = _split_endpoint(_raw_image_endpoint)


class Config:
    # Azure OpenAI text
    AZURE_TEXT_API_KEY = os.getenv("AZURE_TEXT_API_KEY")
    AZURE_TEXT_ENDPOINT = _text_endpoint
    AZURE_TEXT_API_VERSION = os.getenv("AZURE_TEXT_API_VERSION", _text_version or "2024-02-01-preview")
    AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4.1"

    # Azure AI Speech
    AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
    AZURE_SPEECH_ENDPOINT = os.getenv("AZURE_SPEECH_ENDPOINT")
    AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

    # Azure Image
    AZURE_IMAGE_API_KEY = os.getenv("AZURE_IMAGE_KEY")
    AZURE_IMAGE_ENDPOINT = _image_endpoint
    AZURE_IMAGE_API_VERSION = os.getenv("AZURE_IMAGE_API_VERSION", _image_version or "2023-12-01-preview")
    AZURE_IMAGE_DEVELOPMENT_NAME = "FLUX.1-Kontext-pro"

    ARTICLE_DIR = "./article"
    OUTPUT_DIR = "./output"
    JP_FONT_PATH = os.getenv("JP_FONT_PATH")
    MOVIE_DIR = "./movie"

```
## state.pyの作成
```py:state.py
from typing import TypedDict


class ArticleData(TypedDict):
    title: str
    display_title: str
    content: str
    date: str


class AgentState(TypedDict):
    start_date: str
    end_date: str
    run_output_dir: str
    articles: list[ArticleData]
    image_prompts: list[str]
    audio_paths: list[str]
    image_paths: list[str]
    script_paths: list[str]
    video_path: str | None
    error: str | None

```
## nodes.pyの作成
```py:nodes.py
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
            "article_titles": [article['display_title'] for article in target_articles],
            "articles": target_articles
        }
    )

    return {'articles': target_articles, 'run_output_dir': run_dir}


def generate_assets_node(state: AgentState):
    audio_paths = []
    image_paths = []
    image_prompts = []
    script_paths = []
    voice_outputs = []
    image_outputs = []

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

        script_filename = f"script_{i}.txt"
        script_path = os.path.join(run_dir, script_filename)
        with open(script_path, "w", encoding="utf-8") as script_file:
            script_file.write(article['content'])
        script_paths.append(script_path)

        voice_outputs.append({
            "index": i,
            "article_title": article.get('display_title') or article['title'],
            "audio_path": audio_path,
            "script_path": script_path,
            "spoken_text": article['content']
        })

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
        "generate_assets",
        {
            "audio_files": audio_paths,
            "image_files": image_paths,
            "script_files": script_paths,
            "voice_outputs": voice_outputs,
            "image_prompts": image_prompts,
            "image_outputs": image_outputs
        }
    )

    return {
        'audio_paths': audio_paths,
        'image_paths': image_paths,
        'image_prompts': image_prompts,
        'script_paths': script_paths,
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

```
## graph.pyの作成
```py:graph.py
from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import fetch_articles_node, generate_assets_node, create_short_video_node


def create_graph():
    workflow = StateGraph(AgentState)

    # ノードの登録
    workflow.add_node("fetch_articles", fetch_articles_node)
    workflow.add_node("generate_assets", generate_assets_node)
    workflow.add_node("create_video", create_short_video_node)

    # エッジの接続
    workflow.set_entry_point("fetch_articles")
    workflow.add_edge("fetch_articles", "generate_assets")
    workflow.add_edge("generate_assets", "create_video")
    workflow.add_edge("create_video", END)

    return workflow.compile()

```
## main.pyの作成
```py:main.py
import os
import typer
from typing import Annotated
from graph import create_graph
from state import AgentState
from config import Config

app = typer.Typer()


@app.command()
def generate(
    start_date: Annotated[str, typer.Argument(help="開始日 (YYYYMMDD)")],
    end_date: Annotated[str, typer.Argument(help="終了日 (YYYYMMDD)")]
):
    """
    指定した期間(YYYYMMDD)のニュース記事からYouTubeショートを生成します。
    """
    typer.echo(f"🚀 処理を開始: {start_date} から {end_date}")

    # グラフの構築とコンパイル
    graph = create_graph()

    # 初回ステートの初期化
    run_output_dir = os.path.join(
        Config.OUTPUT_DIR, f"{start_date}_{end_date}"
    )
    os.makedirs(run_output_dir, exist_ok=True)

    initial_state: AgentState = {
        "start_date": start_date,
        "end_date": end_date,
        "run_output_dir": run_output_dir,
        "articles": [],
        "image_prompts": [],
        "audio_paths": [],
        "image_paths": [],
        "script_paths": [],
        "video_path": None,
        "error": None
    }

    # LangGraphの実行
    try:
        for output in graph.stream(initial_state):
            for node_name, state_update in output.items():
                typer.echo(f"✅ Node [{node_name}] が完了しました")

        typer.echo(f"✨ 全工程が完了しました！ output/ フォルダを確認してください。")
    except Exception as e:
        typer.secho(f"❌ エラーが発生しました: {e}", fg=typer.colors.RED)


if __name__ == "__main__":
    app()


```
## 動作チェック
```bash
uv run 20260212 20260217
```
## 動画編集
### Canva
### 素材のアップロード
生成した動画と音声をアップロードします
### 動画と音声を合成
タイムラインに動画と音声を置きます
### 文字の挿入
ログにある文言を動画に入れます。
### BGMとSEの合成
BGMとSEを入れます。
###

## Youtubeにアップロード
## まとめ
