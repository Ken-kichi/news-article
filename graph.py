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
