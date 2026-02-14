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
