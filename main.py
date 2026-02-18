import os
import re
from datetime import datetime
import typer
from typing import Annotated
from graph import create_graph
from state import AgentState
from config import Config


def _resolve_run_output_dir(base_path: str) -> str:
    """同名ディレクトリが存在する場合は末尾に ver_n を付けて重複を避ける"""
    if not os.path.exists(base_path):
        return base_path

    version = 1
    while True:
        candidate = f"{base_path}_ver_{version}"
        if not os.path.exists(candidate):
            return candidate
        version += 1


def _extract_article_meta(article_path: str) -> tuple[str, str]:
    """ファイル名から日付とスラグを取得。形式外の場合は当日の日付を利用。"""
    basename = os.path.basename(article_path)
    name_no_ext, _ = os.path.splitext(basename)
    match = re.match(r"(\d{8})_(.+)", name_no_ext)
    if match:
        return match.group(1), match.group(2)
    today = datetime.now().strftime("%Y%m%d")
    return today, name_no_ext or today


def _sanitize_slug(slug: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "_", slug)
    return cleaned.strip("_") or "article"


def _resolve_article_argument(article_arg: str) -> tuple[str, str | None]:
    """
    引数がファイルパスならそのまま返し、8桁の日付なら article/ 内のファイルを探索して返す。
    戻り値は (絶対パス, 推定日付 or None)。
    """
    candidate_path = os.path.abspath(article_arg)
    if os.path.isfile(candidate_path):
        return candidate_path, None

    if not os.path.isabs(article_arg):
        relative_path = os.path.abspath(
            os.path.join(Config.ARTICLE_DIR, article_arg))
        if os.path.isfile(relative_path):
            return relative_path, None

    if re.match(r"^\d{8}$", article_arg):
        date_str = article_arg
        article_dir = Config.ARTICLE_DIR
        if not os.path.isdir(article_dir):
            raise typer.BadParameter("article ディレクトリが存在しません。")
        matches = [
            name for name in os.listdir(article_dir)
            if name.startswith(f"{date_str}_") and name.endswith(".md")
        ]
        if not matches:
            raise typer.BadParameter(f"{date_str} で始まる記事ファイルが見つかりません。")
        if len(matches) > 1:
            raise typer.BadParameter(
                f"{date_str} の記事が複数あります。ファイルパスで指定してください。"
            )
        return os.path.abspath(os.path.join(article_dir, matches[0])), date_str

    raise typer.BadParameter("8桁の日付または記事ファイルパスを指定してください。")


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
    base_output_dir = os.path.join(
        Config.OUTPUT_DIR, f"{start_date}_{end_date}"
    )
    run_output_dir = _resolve_run_output_dir(base_output_dir)
    os.makedirs(run_output_dir, exist_ok=True)

    initial_state: AgentState = {
        "start_date": start_date,
        "end_date": end_date,
        "run_output_dir": run_output_dir,
        "single_article_path": None,
        "articles": [],
        "audio_paths": [],
        "image_paths": [],
        "script_paths": [],
        "thumbnail_path": None,
        "thumbnail_title": None,
        "video_path": None,
        "youtube_metadata_path": None,
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


@app.command("generate-article")
def generate_single_article(
    article_identifier: Annotated[str, typer.Argument(help="記事の8桁日付または記事ファイルパス")],
    date_override: Annotated[str | None, typer.Option(
        help="出力日時 (YYYYMMDD)。省略時は記事情報から推測)")] = None,
):
    """
    単一の記事ファイルからYouTubeショートを生成します。
    """
    article_path, inferred_date = _resolve_article_argument(article_identifier)

    if date_override and not re.match(r"^\d{8}$", date_override):
        raise typer.BadParameter("date は YYYYMMDD 形式で指定してください。")

    date_from_file, slug = _extract_article_meta(article_path)
    date_str = date_override or inferred_date or date_from_file
    safe_slug = _sanitize_slug(slug)

    graph = create_graph()

    base_output_dir = os.path.join(
        Config.OUTPUT_DIR, f"{date_str}_{safe_slug}"
    )
    run_output_dir = _resolve_run_output_dir(base_output_dir)
    os.makedirs(run_output_dir, exist_ok=True)

    initial_state: AgentState = {
        "start_date": date_str,
        "end_date": date_str,
        "run_output_dir": run_output_dir,
        "single_article_path": article_path,
        "articles": [],
        "audio_paths": [],
        "image_paths": [],
        "script_paths": [],
        "thumbnail_path": None,
        "thumbnail_title": None,
        "video_path": None,
        "youtube_metadata_path": None,
        "error": None
    }

    typer.echo(f"🚀 単体記事モードで処理を開始")
    try:
        for output in graph.stream(initial_state):
            for node_name, state_update in output.items():
                typer.echo(f"✅ Node [{node_name}] が完了しました")

        typer.echo(f"✨ 完了: {run_output_dir} を確認してください。")
    except Exception as e:
        typer.secho(f"❌ エラーが発生しました: {e}", fg=typer.colors.RED)


if __name__ == "__main__":
    app()
