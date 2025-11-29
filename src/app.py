"""
Chainlit Frontend for Text2SQL Chatbot with Graph Visualization and LangGraph Debugging
"""

import chainlit as cl
from text2sql_agent import process_question_stream, generate_graph_visualization
import json

##Generate workflow diagram once at module load (optional)
##Uncomment the lines below if you want to generate the diagram:
try:
    workflow_diagram_path = generate_graph_visualization("text2sql_workflow.png")
    if workflow_diagram_path:
        print(f"✅ Workflow diagram generated: {workflow_diagram_path}")
except Exception as e:
    print(f"⚠️ Warning: Could not generate workflow diagram: {e}")

# Set page configuration
@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    
    await cl.Message(
        content="👋 Welcome to the Multi-Agent Assistant!\n\n"
                "I can help you with two main tasks:\n"
                "1. **E-commerce Data Analysis** (SQL): Query the database about orders, customers, products, etc.\n"
                "2. **Vision Browser Support** (RAG): Answer questions from the Vision documentation.\n\n"
                "**Example questions:**\n"
                "- *SQL*: How many orders were delivered?\n"
                "- *SQL*: What are the top 5 product categories by sales?\n"
                "- *RAG*: How do I create a profile in Vision?\n"
                "- *RAG*: What is video spoofing?\n\n"
                "Go ahead and ask me anything! 🚀"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages with debugging visualization"""
    
    user_question = message.content
    
    # Create main processing step
    async with cl.Step(name="🤖 Agent Workflow", type="llm") as workflow_step:
        
        # Dictionary to hold step references
        node_steps = {}
        final_result = None
        
        try:
            # Stream through the agent execution
            async for event in process_question_stream(user_question):
                event_type = event.get("type")
                
                # Handle node start
                if event_type == "node_start":
                    node_name = event["node"]
                    node_display_names = {
                        "guardrails_agent": "🛡️ Guardrails & Routing",
                        "sql_agent": "📝 Generate SQL Query",
                        "execute_sql": "⚙️ Execute SQL Query",
                        "analysis_agent": "💬 Analyze Results",
                        "error_agent": "🔧 Handle Error",
                        "decide_graph_need": "📊 Decide Graph Need",
                        "viz_agent": "📈 Generate Graph",
                        "rag_agent": "📚 Search Documentation"
                    }
                    
                    display_name = node_display_names.get(node_name, node_name)
                    
                    # Create a step for this node
                    node_step = cl.Step(
                        name=display_name,
                        type="tool",
                        parent_id=workflow_step.id
                    )
                    await node_step.send()
                    node_steps[node_name] = node_step
                
                # Handle node end
                elif event_type == "node_end":
                    node_name = event["node"]
                    output = event["output"]
                    state = event["state"]
                    
                    if node_name in node_steps:
                        node_step = node_steps[node_name]
                        
                        # Format output based on node type
                        output_text = ""
                        
                        if node_name == "guardrails_agent":
                            source = output.get("source", "unknown")
                            output_text = f"**Routed to:** {source.upper()}"

                        elif node_name == "sql_agent":
                            sql = output.get("sql_query", "")
                            output_text = f"**Generated SQL Query:**\n```sql\n{sql}\n```"
                        
                        elif node_name == "execute_sql":
                            if output.get("error"):
                                output_text = f"❌ **Error:**\n```\n{output['error']}\n```"
                            else:
                                result = output.get("query_result", "")
                                # Truncate long results for display
                                if len(result) > 500:
                                    result = result[:500] + "\n... (truncated)"
                                output_text = f"**Query Results:**\n```json\n{result}\n```"
                        
                        elif node_name == "analysis_agent":
                            output_text = "✅ **Analysis Complete:** Generated natural language response."
                        
                        elif node_name == "error_agent":
                            corrected = output.get("sql_query", "")
                            iteration = output.get("iteration", 0)
                            output_text = f"**Corrected SQL (Attempt {iteration}):**\n```sql\n{corrected}\n```"
                        
                        elif node_name == "decide_graph_need":
                            needs_graph = output.get("needs_graph", False)
                            graph_type = output.get("graph_type", "")
                            if needs_graph:
                                output_text = f"✅ **Graph Needed:** {graph_type.upper()} chart"
                            else:
                                output_text = "ℹ️ **No graph needed** for this query"
                        
                        elif node_name == "viz_agent":
                            has_graph = bool(output.get("graph_json"))
                            if has_graph:
                                output_text = "✅ Graph generated successfully"
                            else:
                                output_text = "⚠️ Graph generation skipped"
                        
                        elif node_name == "rag_agent":
                            sources = output.get("rag_context", [])
                            output_text = f"✅ **RAG Search Complete**\n\n**Retrieved Sources:**\n" + "\n".join([f"- {s}" for s in sources])
                        
                        # Update the step with output
                        node_step.output = output_text
                        await node_step.update()
                
                # Handle final result
                elif event_type == "final":
                    final_result = event["result"]
                
                # Handle errors
                elif event_type == "error":
                    error_msg = event["error"]
                    workflow_step.output = f"❌ **Error:** {error_msg}"
                    await workflow_step.update()
                    return
            
            # Mark workflow as complete
            workflow_step.output = "✅ Workflow completed successfully"
            await workflow_step.update()
        
        except Exception as e:
            workflow_step.output = f"❌ **Unexpected Error:** {str(e)}"
            await workflow_step.update()
            raise
    
    # Now send the final response outside the workflow step
    if final_result:
        # Build the response content
        # Only show SQL query if it exists and is not empty
        if final_result.get('sql_query') and final_result['sql_query'].strip():
            response_content = f"""**Generated SQL Query:**
```sql
{final_result['sql_query']}
```



**Query Results:**
```json
{final_result.get('query_result', 'No results')}
```

**Answer:**
{final_result['final_answer']}
"""
        elif final_result.get('source') == 'rag':
             response_content = f"""**Answer:**
{final_result['final_answer']}

**Sources:**
""" + "\n".join([f"- {s}" for s in final_result.get('rag_context', [])])
        else:
            # For greetings or out of scope messages, just show the answer
            response_content = final_result['final_answer']
        
        # If there was an error, include it
        if final_result.get('error'):
            response_content += f"\n\n⚠️ **Note:** {final_result['error']}"
        
        # Send text response
        await cl.Message(content=response_content).send()
        
        # Send graph if available - use Chainlit's native Plotly element
        if final_result.get('needs_graph') and final_result.get('graph_json'):
            # Send interactive Plotly visualization using Chainlit's Plotly element
            import plotly.graph_objects as go
            
            # Parse the JSON back to a figure
            fig = go.Figure(json.loads(final_result['graph_json']))
            
            graph_element = cl.Plotly(
                name=f"{final_result.get('graph_type', 'chart')}_visualization",
                figure=fig,
                display="inline"
            )
            
            await cl.Message(
                content=f"📊 **Interactive Visualization ({final_result.get('graph_type', 'chart').title()} Chart)**\n\n*Hover over the chart for details, zoom, and pan!*",
                elements=[graph_element]
            ).send()


@cl.on_chat_end
async def end():
    """Handle chat end"""
    await cl.Message(content="Thanks for using the Text2SQL Assistant! 👋").send()


if __name__ == "__main__":
    # Run with: chainlit run app.py
    pass
