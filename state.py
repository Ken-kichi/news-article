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
    single_article_path: str | None
    articles: list[ArticleData]
    audio_paths: list[str]
    image_paths: list[str]
    script_paths: list[str]
    thumbnail_path: str | None
    thumbnail_title: str | None
    video_path: str | None
    youtube_metadata_path: str | None
    error: str | None
