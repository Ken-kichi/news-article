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
