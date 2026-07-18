from langgraph.graph import StateGraph, END
from backend.app.state import OrchestrationState
from backend.app.nodes import dataset_agent, hyperparam_agent, training_agent, eval_agent, decision_agent, report_agent


def route_decision(state: dict) -> str:
    decision = state.get("decision", "fail")
    iteration = state.get("iteration", 0)
    max_iter = state.get("run_config", {}).get("max_iterations", 3)

    if decision == "stop":
        return "stop"
    if decision == "fail" or iteration >= max_iter:
        return "fail"
    if decision == "adjust":
        return "adjust"
    if decision == "continue":
        return "continue"
    return "fail"


def build_graph(llm=None):
    g = StateGraph(dict)

    g.add_node("dataset_agent", lambda s: dataset_agent.dataset_node(s, llm))
    g.add_node("hyperparam_agent", lambda s: hyperparam_agent.hyperparam_node(s, llm))
    g.add_node("training_agent", training_agent.training_node)
    g.add_node("eval_agent", eval_agent.eval_node)
    g.add_node("decision_agent", lambda s: decision_agent.decision_node(s, llm))
    g.add_node("report_agent", lambda s: report_agent.report_node(s, llm))

    g.set_entry_point("dataset_agent")
    g.add_edge("dataset_agent", "hyperparam_agent")
    g.add_edge("hyperparam_agent", "training_agent")
    g.add_edge("training_agent", "eval_agent")
    g.add_edge("eval_agent", "decision_agent")

    g.add_conditional_edges(
        "decision_agent",
        route_decision,
        {
            "adjust": "hyperparam_agent",
            "continue": "training_agent",
            "stop": "report_agent",
            "fail": "report_agent",
        },
    )

    g.add_edge("report_agent", END)

    return g.compile()
