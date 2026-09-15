"""AgentHub entry point."""

from agent_demos import (
    demo_agent_hub,
    demo_math_agent,
    demo_order_agent,
    demo_supervisor,
    demo_support_agent,
)
from graph_demos import demo_graph, demo_math_agent_graph, demo_order_graph
from langchain_demos import (
    demo_history,
    demo_parsers,
    demo_structured_output,
    demo_toolkit,
    demo_tools,
)
from models import build_model


def main() -> None:
    """Start AgentHub and run the current demos."""
    print("AgentHub started.")

    model = build_model()
    demo_history(model)
    demo_structured_output(model)
    demo_parsers(model)
    demo_tools(model)
    demo_toolkit(model)
    demo_graph(model)
    demo_order_graph(model)
    demo_math_agent(model)
    demo_order_agent(model)
    demo_support_agent(model)
    demo_supervisor(model)
    demo_agent_hub(model)
    demo_math_agent_graph(model)


# Only run main() when this file is executed directly
if __name__ == "__main__":
    main()
