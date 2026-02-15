from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import (
    fetch_articles_node,
    generate_audio_assets_node,
    generate_visual_assets_node,
    create_short_video_node,
    generate_youtube_metadata_node,
)


def create_graph():
    workflow = StateGraph(AgentState)

    # ノードの登録
    workflow.add_node("fetch_articles", fetch_articles_node)
    workflow.add_node("generate_audio_assets", generate_audio_assets_node)
    workflow.add_node("generate_visual_assets", generate_visual_assets_node)
    workflow.add_node("create_video", create_short_video_node)
    workflow.add_node("generate_youtube_metadata",
                      generate_youtube_metadata_node)

    # エッジの接続
    workflow.set_entry_point("fetch_articles")
    workflow.add_edge("fetch_articles", "generate_audio_assets")
    workflow.add_edge("generate_audio_assets", "generate_visual_assets")
    workflow.add_edge("generate_visual_assets", "create_video")
    workflow.add_edge("create_video", "generate_youtube_metadata")
    workflow.add_edge("generate_youtube_metadata", END)

    return workflow.compile()
