import os
import re
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
# from groq import Groq

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

# -----------------------------
# INIT APP
# -----------------------------
app = FastAPI(title="STP AI Backend (Groq + MCP)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# INIT GROQ
# -----------------------------


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

MCP_URL = os.getenv("MCP_URL")

# -----------------------------
# REQUEST MODEL
# -----------------------------
from typing import List, Optional

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str
    history: Optional[List[Message]] = []


# -----------------------------
# 🔥 SYSTEM PROMPT (ARIA)
# -----------------------------
SYSTEM_PROMPT = """
You are **Aria**, an expert AI assistant for a **Sewage Treatment Plant (STP)** monitoring system with live PostgreSQL database access via MCP.

---

## 🔒 IDENTITY & SCOPE

You are STRICTLY an STP plant monitoring assistant.

If the user's question is NOT about: flow, MLD, pumps, tank levels, energy, HLT, or any plant operation data:
→ Respond ONLY with: `⛔ OUT_OF_CONTEXT — I only assist with STP plant monitoring queries.`
→ Do NOT generate SQL. Do NOT try to help. Stop immediately.

---

## 🗄️ DATABASE RULE (NON-NEGOTIABLE)

- Table: `"ATL_MPS"` — ALWAYS. No exceptions.
- NEVER use date-suffixed variants like `"ATL_MPS_31032026"`
- ALL column names MUST be double-quoted

---

## 📋 COLUMN REFERENCE

| Metric | Column |
|---|---|
| Timestamp | `"DateAndTime"` |
| Cumulative Flow (MLD) | `"[PLC]FIT101_TOTAL.D_MLD"` |
| Flow per minute | `"[PLC]FIT101_MIN.PER_MIN"` |
| HLT Tank Level | `"[PLC]HLT101.OUTPUT"` |
| Pump 1–6 Status | `"[PLC]P1.ONOFF"` → `"[PLC]P6.ONOFF"` |
| P1 Energy (KWH) | `"[PLC]P_DATA[18]"` |
| P2 Energy (KWH) | `"[PLC]P_DATA[39]"` |
| P3 Energy (KWH) | `"[PLC]P_DATA[60]"` |
| P4 Energy (KWH) | `"[PLC]P_DATA[81]"` |
| P5 Energy (KWH) | `"[PLC]P_DATA[102]"` |
| P6 Energy (KWH) | `"[PLC]P_DATA[123]"` |

---

## ⚙️ FLOW CALCULATION RULES

**Flow in m³/hr MUST use LAG — NEVER use raw MLD directly.**

```sql
WITH flow_calc AS (
  SELECT
    "DateAndTime",
    "[PLC]FIT101_TOTAL.D_MLD" AS mld,
    LAG("[PLC]FIT101_TOTAL.D_MLD") OVER (ORDER BY "DateAndTime") AS prev_mld
  FROM "ATL_MPS"
  WHERE "DateAndTime" > NOW() - INTERVAL '1 hour'
)
SELECT
  "DateAndTime",
  ROUND(NULLIF((mld - prev_mld) * 60000, 0)::numeric, 2) AS flow_m3_hr
FROM flow_calc
WHERE (mld - prev_mld) > 0   -- negative = meter reset → exclude
ORDER BY "DateAndTime" DESC
LIMIT 20;
```

**LAG Rules:**
- LAG → ONLY for flow calculation
- NEVER apply LAG to energy columns
- NEVER filter on exact timestamp inside a LAG query — use CTE

---

## 🧠 SQL GENERATION RULES

- SELECT only — no INSERT / UPDATE / DELETE / DROP
- Always double-quote every column and table name
- Latest snapshot → `ORDER BY "DateAndTime" DESC LIMIT 1`
- Time windows → use `INTERVAL` (e.g., `NOW() - INTERVAL '24 hours'`)
- Always wrap numerics: `ROUND(value::numeric, 2)`
- Non-aggregated queries → `LIMIT 100` unless user specifies otherwise
- Energy columns → use raw delta (current − previous row), NOT LAG function
- If the request is ambiguous → ASK for clarification before writing SQL

---

## ❗ ANTI-HALLUCINATION RULES

- NEVER assume, guess, or fabricate values
- NEVER invent column names not listed above
- NEVER answer plant questions from memory — ALWAYS execute SQL via MCP tool
- If data is missing or NULL in results → say so explicitly, do not fill in estimates
- If unsure about what the user means → ask: *"Can you confirm what you mean by [term]?"*

---

## 📤 STRICT RESPONSE FORMAT

Every response MUST follow this exact structure:

---

### 🔍 SQL Query

```sql
-- [Brief comment: what this query does]
<your SQL here>
```

---

### 📊 Results

| Column 1 | Column 2 | ... |
|---|---|---|
| value | value | ... |

> *Showing N rows · Queried at [timestamp if available]*

---

### 📈 Operational Summary

| Parameter | Value | Status |
|---|---|---|
| Flow Rate | X MLD | 🟢 Normal / 🔴 High / 🟡 Low |
| Active Pumps | P1, P3 ON · P2, P4, P5, P6 OFF | — |
| HLT Tank Level | X% / X m | 🟢 / 🔴 |
| Energy (last period) | X KWH | — |

---

### 💡 Insights

- **Flow:** [Normal / High / Low — explain why with numbers]
- **Pumps:** [Which are ON/OFF — note any anomaly]
- **Tank:** [Level trend — rising/falling/stable]
- **Energy:** [Consumption trend or anomaly if queried]
- **⚠️ Anomaly:** [Flag resets, spikes, gaps, or suspicious values]

---

## 🚦 STATUS THRESHOLDS (use for 🟢🟡🔴 flags)

| Metric | 🟢 Normal | 🟡 Caution | 🔴 Alert |
|---|---|---|---|
| Flow | Plant-defined range | ±15% of avg | >30% deviation |
| HLT Level | 40–80% | 20–40% or 80–90% | <20% or >90% |
| Active Pumps | 2–4 ON | 1 or 5 ON | 0 or 6 ON |

---

## 🚫 ABSOLUTE RULES (never break these)

1. Always query `FROM "ATL_MPS"`
2. Always show the SQL used
3. Always show results as a table
4. Always include the Operational Summary table
5. Never guess data — if no rows returned, say: *"No data found for this period."*
6. Never answer STP questions without running SQL first.
"""
# -----------------------------
# 🧹 CLEAN SQL OUTPUT
# -----------------------------
def clean_sql(text):
    if text is None:
        return ""
    text = re.sub(r"```sql|```", "", text)
    return text.strip()


# -----------------------------
# 🧠 GENERATE SQL (GROQ)
# -----------------------------
def generate_sql(user_query: str, history: List[Message]):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})
        
    user_prompt = f"""
User Question:
{user_query}

CRITICAL INSTRUCTION:
- If this question is NOT about STP plant operations (pumps, flow, tanks, energy, MLD), return ONLY the word: OUT_OF_CONTEXT
- If it IS about STP plant operations, return ONLY the SQL query. No explanation, no markdown.
"""
    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        messages=messages,
        model="openai/gpt-oss-120b",
        temperature=0.1
    )
    return clean_sql(response.choices[0].message.content)


# -----------------------------
# 🧠 GENERATE INSIGHTS (GROQ)
# -----------------------------
def generate_insights(user_query, sql, data):
    prompt = f"""
User Question: {user_query}

SQL Used:
{sql}

Data:
{data}

Provide a professional operational summary for the STP engineers.
Use the following format:

### 📈 Operational Status
*   **Flow:** [Status here]
*   **Pumps:** [Which are ON/OFF]
*   **Tank Levels:** [Observation]

### 🔍 Key Observations
[One or two bullet points about anomalies or efficiency]

Keep it structured and use bold text for key metrics.
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    return clean_sql(response.choices[0].message.content)


# -----------------------------
# 🔐 SQL VALIDATION
# -----------------------------
def validate_sql(sql: str):
    if not sql:
        raise HTTPException(status_code=400, detail="Generated SQL is empty or None")
    # Strip whitespace and common SQL comments at the start
    clean_sql = re.sub(r'^\s*(--.*?\n|/\*.*?\*/)', '', sql, flags=re.DOTALL | re.MULTILINE).strip().lower()

    # Allow SELECT and WITH (for CTEs)
    if not (clean_sql.startswith("select") or clean_sql.startswith("with")):
        raise HTTPException(status_code=400, detail="Only SELECT or WITH queries allowed")

    forbidden = ["drop ", "delete ", "update ", "insert ", "alter "]
    for word in forbidden:
        if word in clean_sql:
            raise HTTPException(status_code=400, detail=f"Forbidden keyword detected: {word.strip().upper()}")

    if '"ATL_MPS"' not in sql:
        raise HTTPException(status_code=400, detail="Security: Query must target 'ATL_MPS' table.")


# -----------------------------
# 🔌 CALL MCP
# -----------------------------
import json

def run_mcp_query(sql: str):
    """
    Handles the MCP SSE handshake:
    1. Connects to the SSE endpoint to get a session URL.
    2. Posts the JSON-RPC tool call.
    3. Waits for the response on the SSE stream.
    """
    try:
        print(f"Connecting to MCP SSE: {MCP_URL}")
        with requests.get(MCP_URL, stream=True, timeout=30) as r:
            session_url = None
            initialized = False
            
            for line in r.iter_lines():
                if not line:
                    continue
                
                decoded = line.decode('utf-8')
                
                # Step 1: Get the session endpoint
                if decoded.startswith("data: ") and session_url is None:
                    endpoint = decoded[6:].strip()
                    if endpoint.startswith("http"):
                        session_url = endpoint
                    else:
                        base = MCP_URL.split("/sse")[0]
                        session_url = f"{base}{endpoint}"
                    
                    print(f"Session URL found: {session_url}")
                    
                    # Step 2: Send INITIALIZE request
                    init_payload = {
                        "jsonrpc": "2.0",
                        "id": "init_1",
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "stp-ai-backend", "version": "1.0.0"}
                        }
                    }
                    requests.post(session_url, json=init_payload, timeout=5)
                    continue

                if decoded.startswith("data: "):
                    try:
                        message = json.loads(decoded[6:])
                        
                        # Step 3: Handle Initialize Response & Send Initialized Notification
                        if message.get("id") == "init_1":
                            print("MCP Initialized. Sending notification...")
                            notif_payload = {
                                "jsonrpc": "2.0",
                                "method": "notifications/initialized"
                            }
                            requests.post(session_url, json=notif_payload, timeout=5)
                            
                            # Step 4: Now call the TOOL
                            print(f"Calling tool: execute_sql")
                            tool_payload = {
                                "jsonrpc": "2.0",
                                "id": "call_1",
                                "method": "tools/call",
                                "params": {
                                    "name": "execute_sql",
                                    "arguments": {"sql": sql}
                                }
                            }
                            requests.post(session_url, json=tool_payload, timeout=5)
                            initialized = True
                            continue
                        
                        # Step 5: Listen for the TOOL result
                        if message.get("id") == "call_1":
                            result = message.get("result")
                            if result and "content" in result:
                                content = result["content"]
                                if content and len(content) > 0:
                                    text_data = content[0].get("text", "")
                                    try:
                                        return json.loads(text_data)
                                    except:
                                        return text_data
                            return result
                        
                        elif message.get("id") == "call_1" and "error" in message:
                            raise Exception(f"MCP Tool Error: {message['error']}")
                            
                    except json.JSONDecodeError:
                        continue
            
        raise Exception("Stream closed without receiving response")

    except Exception as e:
        print(f"MCP Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"MCP Error: {str(e)}")


# -----------------------------
# 🧠 GENERATE INSIGHTS (GROQ)
# -----------------------------
def generate_insights(user_query, sql, data):
    prompt = f"""
User Question: {user_query}

SQL Used:
{sql}

Data:
{data}

Give:
- Flow condition
- Pump status
- Tank level
- Any anomalies

Keep it short.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    content = response.choices[0].message.content
    return content.strip() if content else "No insights available."


# -----------------------------
# 🚀 MAIN API
# -----------------------------
@app.post("/chat")
def chat(req: QueryRequest):
    try:
        # 1. Generate SQL (with context check)
        sql = generate_sql(req.query, req.history)

        # 2. Check for out-of-context refusal
        if "OUT_OF_CONTEXT" in sql.upper():
            return {
                "query": req.query,
                "sql": "N/A",
                "data": None,
                "insights": "I am **Aria**, your STP Operational Assistant. I can only assist with questions related to plant operations, flow, pumps, and energy consumption. How can I help you with the plant today?"
            }

        # 3. Validate and Modify SQL (Retry Loop)
        max_retries = 3
        data = None
        for attempt in range(max_retries):
            try:
                validate_sql(sql)
                data = run_mcp_query(sql)
                break  # Success
            except Exception as e:
                error_msg = str(e.detail) if hasattr(e, "detail") else str(e)
                if attempt == max_retries - 1:
                    if isinstance(e, HTTPException):
                        raise e
                    raise HTTPException(status_code=500, detail=error_msg)
                
                # Ask the model to fix it
                correction_prompt = f"The generated SQL has an error:\n{error_msg}\n\nPlease fix the following SQL query and return ONLY the corrected SQL. Do not include markdown or explanations. Make sure it targets the 'ATL_MPS' table.\nSQL: {sql}"
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": req.query},
                        {"role": "assistant", "content": sql},
                        {"role": "user", "content": correction_prompt}
                    ],
                    temperature=0.1
                )
                sql = clean_sql(response.choices[0].message.content)

        # 5. Generate Insights
        insights = generate_insights(req.query, sql, data)

        return {
            "query": req.query,
            "sql": sql,
            "data": data,
            "insights": insights
        }

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# 📂 SERVE FRONTEND
# -----------------------------
static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/", StaticFiles(directory=static_path, html=True), name="static")