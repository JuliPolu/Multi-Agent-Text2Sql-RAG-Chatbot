"""
Text2SQL Agent using LangGraph
This module implements an Agentic system for converting natural language to SQL queries and also generates graphs.
"""

import os
import sqlite3
from typing import TypedDict
from langgraph.graph import StateGraph, END
from vllm_manager import get_vllm_manager
from monitoring import monitor
import json
import pandas as pd
from rag_agent import rag_node

# Initialize vLLM Manager
vllm_manager = get_vllm_manager()

# Database configuration
DB_PATH = "ecommerce.db"

# Database schema information
SCHEMA_INFO = """
Database Schema for E-commerce System:

1. customers
   - customer_id (TEXT): Unique customer identifier
   - customer_unique_id (TEXT): Unique customer identifier across datasets
   - customer_zip_code_prefix (INTEGER): Customer zip code
   - customer_city (TEXT): Customer city
   - customer_state (TEXT): Customer state

2. orders
   - order_id (TEXT): Unique order identifier
   - customer_id (TEXT): Foreign key to customers
   - order_status (TEXT): Order status (delivered, shipped, etc.)
   - order_purchase_timestamp (TEXT): When the order was placed
   - order_approved_at (TEXT): When payment was approved
   - order_delivered_carrier_date (TEXT): When order was handed to carrier
   - order_delivered_customer_date (TEXT): When customer received the order
   - order_estimated_delivery_date (TEXT): Estimated delivery date

3. order_items
   - order_id (TEXT): Foreign key to orders
   - order_item_id (INTEGER): Item sequence number within order
   - product_id (TEXT): Foreign key to products
   - seller_id (TEXT): Foreign key to sellers
   - shipping_limit_date (TEXT): Shipping deadline
   - price (REAL): Item price
   - freight_value (REAL): Shipping cost

4. order_payments
   - order_id (TEXT): Foreign key to orders
   - payment_sequential (INTEGER): Payment sequence number
   - payment_type (TEXT): Payment method (credit_card, boleto, etc.)
   - payment_installments (INTEGER): Number of installments
   - payment_value (REAL): Payment amount

5. order_reviews
   - review_id (TEXT): Unique review identifier
   - order_id (TEXT): Foreign key to orders
   - review_score (INTEGER): Review score (1-5)
   - review_comment_title (TEXT): Review title
   - review_comment_message (TEXT): Review message
   - review_creation_date (TEXT): When review was created
   - review_answer_timestamp (TEXT): When review was answered

6. products
   - product_id (TEXT): Unique product identifier
   - product_category_name (TEXT): Product category (in Portuguese)
   - product_name_lenght (REAL): Product name length
   - product_description_lenght (REAL): Product description length
   - product_photos_qty (REAL): Number of product photos
   - product_weight_g (REAL): Product weight in grams
   - product_length_cm (REAL): Product length in cm
   - product_height_cm (REAL): Product height in cm
   - product_width_cm (REAL): Product width in cm

7. sellers
   - seller_id (TEXT): Unique seller identifier
   - seller_zip_code_prefix (INTEGER): Seller zip code
   - seller_city (TEXT): Seller city
   - seller_state (TEXT): Seller state

8. geolocation
   - geolocation_zip_code_prefix (INTEGER): Zip code prefix
   - geolocation_lat (REAL): Latitude
   - geolocation_lng (REAL): Longitude
   - geolocation_city (TEXT): City name
   - geolocation_state (TEXT): State code

9. product_category_name_translation
   - product_category_name (TEXT): Category name in Portuguese
   - product_category_name_english (TEXT): Category name in English
"""


class AgentState(TypedDict):
    """State of the agent workflow"""
    question: str
    sql_query: str
    query_result: str
    final_answer: str
    error: str
    iteration: int
    needs_graph: bool
    graph_type: str
    graph_json: str  # Plotly figure JSON for Chainlit
    is_in_scope: bool  # Whether the question is about e-commerce data
    source: str # 'sql' or 'rag'
    rag_context: list # List of source files used for RAG


# Agent configurations with different roles and personalities

AGENT_CONFIGS = {
    "guardrails_agent": {
        "role": "Router and Scope Manager",
        "system_prompt": "You are a router system for a chatbot that has access to two data sources: 1) An E-commerce Database (SQL) and 2) Vision Browser Documentation (RAG). Your job is to route the user's question to the correct agent.",
    },
    "sql_agent": {
        "role": "SQL Expert", 
        "system_prompt": "You are a senior SQL developer specializing in e-commerce databases. Generate only valid SQLite queries without any formatting or explanation.",
    },
    "analysis_agent": {
        "role": "Data Analyst",
        "system_prompt": "You are a helpful data analyst that explains database query results in natural language with clear insights.",
    },
    "viz_agent": {
        "role": "Visualization Specialist", 
        "system_prompt": "You are a data visualization expert. Generate clean, executable Plotly code without any markdown formatting or explanations.",
    },
    "error_agent": {
        "role": "Error Recovery Specialist",
        "system_prompt": "You diagnose and fix SQL errors with expert knowledge of database schemas and query optimization.",
    }
}


@monitor.trace(name="guardrails_agent")
def guardrails_agent(state: AgentState) -> AgentState:
    """Route the question to the appropriate agent (SQL or RAG)"""
    question = state["question"]
    
    prompt = f"""You are a router system. Your job is to determine if a user's question should be answered by the SQL Agent (E-commerce data) or the RAG Agent (Vision Browser Documentation), or if it's a greeting/out of scope.

    DATA SOURCE 1: E-commerce Database (SQL)
    - Customers, orders, products, sellers, payments, reviews (2016-2018)
    - Questions about sales, statistics, locations, order status.

    DATA SOURCE 2: Vision Browser Documentation (RAG)
    - "Vision" is a browser automation tool.
    - Questions about:
        - Profiles, folders, proxies, tags, statuses
        - API reference, automation, extensions, bookmarks
        - Installation, migration, settings, teamwork
        - Video spoofing, cookie robot, mass creation
    
    Examples of SQL (E-commerce) questions:
    - "How many orders were placed last month?"
    - "Top selling products"
    - "Average order value"

    Examples of RAG (Vision) questions:
    - "How do I create a profile?"
    - "What is the API endpoint for creating a folder?"
    - "How to use the cookie robot?"
    - "Explain video spoofing"
    - "How to install Vision?"

    Examples of GREETING messages:
    - "Hi", "Hello", "How are you?"

    User Question: {question}

    Analyze the question and respond in JSON format:
    {{
        "source": "sql" or "rag" or "greeting" or "out_of_scope",
        "reason": "brief explanation"
    }}
    """

    response_content = vllm_manager.generate_response(
        messages=[
            {"role": "system", "content": AGENT_CONFIGS["guardrails_agent"]["system_prompt"]},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response_content)
    source = result.get("source", "out_of_scope")
    state["source"] = source
    
    # If it's a greeting, provide a welcome message
    if source == "greeting":
        state["final_answer"] = "Hi! I am your assistant. I can help you with:\n1. E-commerce data analysis (SQL)\n2. Vision Browser documentation (RAG)\n\nHow can I help you today?"
        return state
    
    # If out of scope
    if source == "out_of_scope":
        state["final_answer"] = "I apologize, but your question appears to be out of scope. I can only answer questions about the E-commerce database or Vision Browser documentation."
    
    return state


@monitor.trace(name="sql_agent")
def sql_agent(state: AgentState) -> AgentState:
    """Generate SQL query from natural language question"""
    question = state["question"]
    iteration = state.get("iteration", 0)
    
    prompt = f"""You are a SQL expert. Convert the following natural language question into a valid SQLite query.

{SCHEMA_INFO}

Question: {question}

Important Guidelines:
1. Use only the tables and columns mentioned in the schema
2. Use proper JOIN clauses when querying multiple tables
3. Return ONLY the SQL query without any explanation or markdown formatting
4. If the question contains multiple sub-questions, generate separate SQL queries separated by semicolons
5. Use aggregate functions (COUNT, SUM, AVG, etc.) appropriately
6. Add LIMIT clauses for queries that might return many rows (default LIMIT 10 unless user specifies)
7. Use proper WHERE clauses to filter data
8. For date comparisons, remember the dates are stored as TEXT in ISO format
9. Each SQL statement should be on its own line for clarity when multiple queries are needed
10. The dataset covers 2016-2018. For questions about "recent", "last month", "this year", etc., use the date '2018-08-31' as "today" or query relative to the maximum order date (e.g., using a subquery like (SELECT MAX(order_purchase_timestamp) FROM orders)).

Generate the SQL query:"""

    sql_query = vllm_manager.generate_response(
        messages=[
            {"role": "system", "content": AGENT_CONFIGS["sql_agent"]["system_prompt"]},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    ).strip()
    # Remove markdown code blocks if present
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
    
    state["sql_query"] = sql_query
    state["iteration"] = iteration + 1
    
    return state


def execute_sql(state: AgentState) -> AgentState:
    """Execute the generated SQL query (handles multiple queries if present)"""
    sql_query = state["sql_query"]
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Split multiple SQL statements (separated by semicolons)
        # Remove empty statements and strip whitespace
        sql_statements = [stmt.strip() for stmt in sql_query.split(';') if stmt.strip()]
        
        all_results = []
        
        # Execute each statement separately
        for i, statement in enumerate(sql_statements):
            cursor.execute(statement)
            
            # Fetch results for this statement
            results = cursor.fetchall()
            
            if results:
                column_names = [description[0] for description in cursor.description]
                
                # Convert to list of dictionaries
                formatted_results = []
                for row in results[:100]:  # Limit to 100 rows per query
                    formatted_results.append(dict(zip(column_names, row)))
                
                # If multiple queries, label them
                if len(sql_statements) > 1:
                    all_results.append({
                        f"query_{i+1}": formatted_results,
                        f"query_{i+1}_sql": statement
                    })
                else:
                    all_results = formatted_results
        
        conn.close()
        
        # Format results
        if not all_results:
            state["query_result"] = "No results found."
        else:
            state["query_result"] = json.dumps(all_results, indent=2)
        
        state["error"] = ""
        
    except Exception as e:
        state["error"] = f"SQL Execution Error: {str(e)}"
        state["query_result"] = ""
    
    return state

@monitor.trace(name="error_agent")
def error_agent(state: AgentState) -> AgentState:
    """Handle errors and attempt to fix the SQL query"""
    error = state["error"]
    sql_query = state["sql_query"]
    question = state["question"]
    iteration = state.get("iteration", 0)
    
    # If we've tried too many times, give up
    if iteration > 3:
        state["final_answer"] = f"I apologize, but I'm having trouble generating a correct SQL query for your question. Error: {error}"
        return state
    
    prompt = f"""The following SQL query failed with an error. Please fix it.

{SCHEMA_INFO}

Original Question: {question}

Failed SQL Query: {sql_query}

Error: {error}

Generate a corrected SQL query that will work. Return ONLY the SQL query without any explanation or markdown formatting:"""

    corrected_query = vllm_manager.generate_response(
        messages=[
            {"role": "system", "content": AGENT_CONFIGS["error_agent"]["system_prompt"]},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    ).strip()
    corrected_query = corrected_query.replace("```sql", "").replace("```", "").strip()
    
    state["sql_query"] = corrected_query
    state["error"] = ""  # Clear the error for retry
    state["iteration"] = iteration + 1  # Increment iteration counter
    
    return state


@monitor.trace(name="analysis_agent")
def analysis_agent(state: AgentState) -> AgentState:
    """Generate natural language answer from query results"""
    question = state["question"]
    sql_query = state["sql_query"]
    query_result = state["query_result"]
    
    prompt = f"""You are a helpful assistant that explains database query results in natural language.

Original Question: {question}

SQL Query Used: {sql_query}

Query Results:
{query_result}

Please provide a clear, concise answer to the original question based on the query results.
Format the answer in a user-friendly way. If the results contain numbers, present them clearly.
If there are multiple queries/results (for multi-part questions), address each part of the question separately.
Use bullet points or numbered lists for multiple answers.

Answer:"""

    final_answer = vllm_manager.generate_response(
        messages=[
            {"role": "system", "content": AGENT_CONFIGS["analysis_agent"]["system_prompt"]},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    ).strip()
    state["final_answer"] = final_answer
    
    return state


@monitor.trace(name="decide_graph_need")
def decide_graph_need(state: AgentState) -> AgentState:
    """Decide if a graph visualization would be helpful for the query"""
    question = state["question"]
    query_result = state["query_result"]
    
    # If no results or error, no graph needed
    if not query_result or query_result == "No results found." or state.get("error"):
        state["needs_graph"] = False
        state["graph_type"] = ""
        return state
    
    prompt = f"""Analyze the following question and query results to determine if a graph visualization would be helpful.

Question: {question}

Query Results Sample:
{query_result[:500]}...

Determine:
1. Would a graph be helpful for this data? (YES/NO)
2. If yes, what type of graph? (bar, line, pie, scatter)

Consider:
- Trends over time → line chart
- Comparisons between categories → bar chart
- Proportions/percentages → pie chart
- Correlations → scatter plot
- Simple counts or single values → NO graph needed

Respond in JSON format:
{{"needs_graph": true/false, "graph_type": "bar/line/pie/scatter/none", "reason": "brief explanation"}}
"""

    response_content = vllm_manager.generate_response(
        messages=[
            {"role": "system", "content": "You are a data visualization expert. Analyze queries and determine if visualization would add value."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    decision = json.loads(response_content)
    state["needs_graph"] = decision.get("needs_graph", False)
    state["graph_type"] = decision.get("graph_type", "none")
    
    return state


@monitor.trace(name="viz_agent")
def viz_agent(state: AgentState) -> AgentState:
    """Generate a graph visualization from query results using LLM-generated Plotly code"""
    query_result = state["query_result"]
    graph_type = state["graph_type"]
    question = state["question"]
    
    try:
        # Parse query results
        results = json.loads(query_result)
        if not results or len(results) == 0:
            state["graph_json"] = ""
            return state
        
        # Convert to DataFrame for context
        df = pd.DataFrame(results)
        columns = df.columns.tolist()
        sample_data = df.head(5).to_dict('records')
        
        # Generate Plotly code using LLM
        prompt = f"""Generate Python code using Plotly to visualize the following data.

Question: {question}
Graph Type: {graph_type}
Columns: {columns}
Sample Data (first 5 rows): {json.dumps(sample_data, indent=2)}
Total Rows: {len(df)}

Requirements:
1. Use plotly.graph_objects or plotly.express
2. The data is already loaded as 'df' (a pandas DataFrame)
3. Create an appropriate {graph_type} chart
4. Limit data to top 20 rows if there are many rows
5. Add proper titles, labels, and formatting
6. The figure variable must be named 'fig'
7. Return ONLY the Python code, no explanations or markdown
8. Do NOT include any import statements
9. Do NOT include code to show the figure (no fig.show())
10. Make the visualization visually appealing with appropriate colors and layout
11. Update the layout for better interactivity (hover info, responsive sizing)

Generate the Plotly code:"""

        plotly_code = vllm_manager.generate_response(
            messages=[
                {"role": "system", "content": AGENT_CONFIGS["viz_agent"]["system_prompt"]},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        ).strip()
        # Remove markdown code blocks if present
        plotly_code = plotly_code.replace("```python", "").replace("```", "").strip()
        
        # Prepare execution environment
        exec_globals = {
            'df': df,
            'pd': pd,
            'json': json
        }
        
        # Import plotly dynamically
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            exec_globals['go'] = go
            exec_globals['px'] = px
        except ImportError:
            print("Plotly not installed. Installing...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'plotly'])
            import plotly.graph_objects as go
            import plotly.express as px
            exec_globals['go'] = go
            exec_globals['px'] = px
        
        # Execute the generated code
        exec(plotly_code, exec_globals)
        
        # Get the figure object
        fig = exec_globals.get('fig')
        
        if fig is None:
            raise ValueError("Generated code did not create a 'fig' variable")
        
        # Export figure as JSON for Chainlit's Plotly element
        graph_json = fig.to_json()
        state["graph_json"] = graph_json
        
    except Exception as e:
        print(f"Graph generation error: {e}")
        print(f"Generated code:\n{plotly_code if 'plotly_code' in locals() else 'No code generated'}")
        state["graph_json"] = ""
    
    return state








def should_retry(state: AgentState) -> str:
    """Decide whether to retry after an error"""
    if state.get("error"):
        iteration = state.get("iteration", 0)
        if iteration <= 3:
            return "retry"
        else:
            return "end"
    return "success"


def should_generate_graph(state: AgentState) -> str:
    """Decide whether to generate a graph"""
    if state.get("needs_graph", False):
        return "viz_agent"
    return "skip_graph"


def check_scope(state: AgentState) -> str:
    """Check source to route to correct agent"""
    source = state.get("source", "out_of_scope")
    if source == "sql":
        return "sql_agent"
    elif source == "rag":
        return "rag_agent"
    else:
        return "end"


# Build the LangGraph workflow
def create_text2sql_graph():
    """Create the LangGraph state graph for Text2SQL with graph generation"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("guardrails_agent", guardrails_agent)
    workflow.add_node("sql_agent", sql_agent)
    workflow.add_node("execute_sql", execute_sql)
    workflow.add_node("analysis_agent", analysis_agent)
    workflow.add_node("error_agent", error_agent)
    workflow.add_node("decide_graph_need", decide_graph_need)
    workflow.add_node("viz_agent", viz_agent)
    workflow.add_node("rag_agent", rag_node)
    
    # Add edges - start with guardrails check
    workflow.set_entry_point("guardrails_agent")
    
    # Conditional edge from guardrails - only proceed if in scope
    workflow.add_conditional_edges(
        "guardrails_agent",
        check_scope,
        {
            "sql_agent": "sql_agent",
            "rag_agent": "rag_agent",
            "end": END
        }
    )
    
    workflow.add_edge("rag_agent", END)
    
    workflow.add_edge("sql_agent", "execute_sql")
    
    # Conditional edge based on execution success
    workflow.add_conditional_edges(
        "execute_sql",
        should_retry,
        {
            "success": "analysis_agent",
            "retry": "error_agent",
            "end": "analysis_agent"
        }
    )
    
    workflow.add_edge("error_agent", "execute_sql")
    workflow.add_edge("analysis_agent", "decide_graph_need")
    
    # Conditional edge for graph generation
    workflow.add_conditional_edges(
        "decide_graph_need",
        should_generate_graph,
        {
            "viz_agent": "viz_agent",
            "skip_graph": END
        }
    )
    
    workflow.add_edge("viz_agent", END)
    
    return workflow.compile()


# Create the compiled graph
text2sql_graph = create_text2sql_graph()



def generate_graph_visualization(output_path: str = "text2sql_workflow.png") -> str:
    """
    Generate a PNG visualization of the LangGraph workflow.
    
    Args:
        output_path: Path where the PNG file will be saved (default: "text2sql_workflow.png")
    
    Returns:
        str: Path to the generated PNG file
    """
    try:
        # Get the graph visualization
        graph_image = text2sql_graph.get_graph().draw_mermaid_png()
        
        # Save to file
        with open(output_path, "wb") as f:
            f.write(graph_image)
        
        print(f"Graph visualization saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error generating graph visualization: {e}")
        print("Make sure you have 'pygraphviz' or 'grandalf' installed:")
        print("  pip install pygraphviz")
        print("  or")
        print("  pip install grandalf")
        return None




async def process_question_stream(question: str):
    """
    Process a natural language question and stream node execution events.
    This is an async generator that yields node events for debugging visualization.
    
    Yields:
        dict: Event with type ('node_start', 'node_end', 'error', 'final') and data
    """
    initial_state = AgentState(
        question=question,
        sql_query="",
        query_result="",
        final_answer="",
        error="",
        iteration=0,
        needs_graph=False,
        graph_type="",
        graph_json="",
        is_in_scope=True,
        source=""
    )
    
    current_state = initial_state.copy()
    
    try:
        # Stream events from the graph
        async for event in text2sql_graph.astream_events(
            initial_state,
            config={"recursion_limit": 50},
            version="v1"
        ):
            event_type = event.get("event")
            
            # Node start event
            if event_type == "on_chain_start":
                node_name = event.get("name", "")
                if node_name in ["guardrails_agent", "sql_agent", "execute_sql", "analysis_agent", 
                               "error_agent", "decide_graph_need", "viz_agent", "rag_agent"]:
                    yield {
                        "type": "node_start",
                        "node": node_name,
                        "input": current_state
                    }
            
            # Node end event
            elif event_type == "on_chain_end":
                node_name = event.get("name", "")
                if node_name in ["guardrails_agent", "sql_agent", "execute_sql", "analysis_agent", 
                               "error_agent", "decide_graph_need", "viz_agent", "rag_agent"]:
                    output = event.get("data", {}).get("output", {})
                    if output:
                        current_state.update(output)
                        yield {
                            "type": "node_end",
                            "node": node_name,
                            "output": output,
                            "state": current_state.copy()
                        }
        
        # Send final result
        yield {
            "type": "final",
            "result": current_state
        }
        
    except Exception as e:
        yield {
            "type": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    # Test the agent
    print("=" * 80)
    print("Text2SQL Agent - Use 'chainlit run app.py' to start the web interface")
    print("=" * 80)
    print("\nThis module is meant to be imported and used via the Chainlit app.")
    print("Run: chainlit run app.py")

