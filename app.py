import requests
from flask_talisman import Talisman
from flask import Flask, render_template, request, jsonify
import warnings
warnings.filterwarnings("ignore")
import os
from dotenv import load_dotenv
from flask_cors import CORS, cross_origin
from functools import wraps
from markupsafe import escape
import openai
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI
import getpass
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
import re
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from flask import g
##**********************************************************ERM packages******************************************************
from langchain.retrievers import EnsembleRetriever
from sqlalchemy import create_engine
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from openai import AzureOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, SystemMessage
import oracledb
from datetime import datetime
## This is a portfolio / demo version of a production RAG chatbot.
## All company-specific names, domains, internal file paths, and credentials
## have been replaced with generic placeholders. Populate the .env file with
## your own values before running.

# Oracle config from .env
ORACLE_CONFIG = {
    "user": os.getenv("ORACLE_USER"),
    "password": os.getenv("ORACLE_PASSWORD"),
    "dsn": oracledb.makedsn(
        os.getenv("ORACLE_HOST"),
        int(os.getenv("ORACLE_PORT")),
        service_name=os.getenv("ORACLE_SERVICE")

    )
}

def get_oracle_connection():
    """Create and return an Oracle DB connection."""
    return oracledb.connect(
        user=ORACLE_CONFIG["user"],
        password=ORACLE_CONFIG["password"],
        dsn=ORACLE_CONFIG["dsn"]
    )

# Initialize Flask app
app = Flask(__name__)

# NOTE: Replace these with your own frontend domain(s). Using env vars keeps
# real deployment domains out of source control.
FINANCE_APP_ORIGIN = os.getenv("FINANCE_APP_ORIGIN", "https://your-frontend-domain.example.com")
ERM_APP_ORIGIN = os.getenv("ERM_APP_ORIGIN", "https://your-erm-frontend.example.com")

cors = CORS(app, resources={
                                r"/get": {"origins": FINANCE_APP_ORIGIN},
                                r"/geterm": {"origins": ERM_APP_ORIGIN}

                            }, methods=["POST"])

azure_api_key = os.getenv('AZURE_API_KEY')
azure_base_url = os.getenv('AZURE_BASE_URL')

# Initialize LLM and Embeddings
llm = AzureChatOpenAI(
    streaming=True,
    model="gpt-4o-mini",
    openai_api_key=azure_api_key,
    api_version="2024-08-01-preview",
    openai_api_base=azure_base_url,
    default_headers={"Ocp-Apim-Subscription-Key": azure_api_key},
    temperature=0

)
ERM_llm = AzureChatOpenAI(
    streaming=True,
    model="gpt-4o-mini",
    openai_api_key=azure_api_key,
    api_version="2024-08-01-preview",
    openai_api_base=azure_base_url,
    default_headers={"Ocp-Apim-Subscription-Key": azure_api_key},
    temperature=0.000001
)


embeddings = AzureOpenAIEmbeddings(
    model="embedding",
    openai_api_key=azure_api_key,
    api_version="2024-08-01-preview",
    openai_api_base=azure_base_url,
    default_headers={"Ocp-Apim-Subscription-Key": azure_api_key},
    show_progress_bar="True"
)

llm_agent = AzureChatOpenAI(
    streaming=True,
    model="chat",
    openai_api_key=azure_api_key,
    api_version="2024-08-01-preview",
    openai_api_base=azure_base_url,
    default_headers={"Ocp-Apim-Subscription-Key": azure_api_key}
    # temperature=0.000001
)

# Initialize collection names
collection_names = [
    "Finance_Batch1",
    "Finance_Batch2",
    "Finance_Batch3",
    "Finance_Batch4",
    "Finance_Batch5",
    "Finance_Batch6",
    "Finance_Batch7",
    "Finance_Batch8",

]

# Persistent vector store dictionary
# NOTE: point this at a local/relative path (or an env var) rather than a
# machine-specific absolute path.
FINANCE_DB_PATH = os.getenv("FINANCE_DB_PATH", "./data/finance_vector_db")

vector_stores = {
    collection_name: Chroma(
        collection_name=collection_name,
        persist_directory=FINANCE_DB_PATH,
        embedding_function=embeddings,
    )
    for collection_name in collection_names
}
# Function to create a state-independent retrieval chain
def create_state_independent_retrieval_chain(collection_name, query):
    # Base prompt for the system
    base_prompt = """   
Role: Answer the question accurately and comprehensively.
Your task is to provide the answer based on the Finance policies vector database you are connected to. Don't generate the answer
if the answer is not from the database provided to you.
Look in-depth and then provide the answer. Don't generate any random answer and don't give any suggestions.
If you don't know the answer, take some time to look more in-depth in the database then give the answer. If user greet's by hi, hello, or saying something else which context is greting then directly greet them back
Restrictions: 
    1) Don't mention in your response about the company vector database or any related keyword.
    2) Say sorry if the answer is out of the vector database and don't give any suggestions.
    3) Scan the entire database only then give the answer.
    4) You have to provide the answer from the database only for each question the user has asked you but in your response don't include Vector database keyword you are restict to use vector database.
    5)  . In case the user asks what kind of questions they can ask, provide them with 5 scenario-based example questions from the vector database itself for which you can answer. Ensure these examples are from the database only which in case if user ask you can provide answer as well.. 
    Scenario-based Example Questions this is how you have to desgin the questions:I'm reviewing the HUB processes as part of a SOX review. I'm not sure if a cycle count needs to be carried out for the Hub I'm reviewing. The value of inventory in the hub is 2.5Mn. Can you suggest if a cycle count is required for the HUB?
    Only in case if possible create question in this format for suggestion but first go thourgh the document and then genrate questions.
    Numbers-based: "Based on the data from the last audit, the total value of inventory discrepancies identified was $500,000. What steps should be taken to address these discrepancies?"
Days-based: "According to the vector database, the last physical inventory count was conducted 180 days ago. Is it necessary to schedule another count soon?"
Case-based: "The vector database indicates that 10% of the inventory items were obsolete during a recent audit. What actions should be taken to manage obsolete inventory?"
Numbers-based: "The variance between the book inventory and physical inventory, as per the vector database, is $250,000. How should this variance be investigated and resolved?"
Days-based: "The cycle count frequency for high-value items is currently set at 90 days according to the vector database. Should this frequency be adjusted based on recent audit findings?"

Ensure these examples are based on numbers, days, and case-based scenarios, aligned with a 20+ years experienced internal auditor's level of expertise. Do not include the example provided above.




    6) If a user greets you, greet them back. If the user gives feedback, respond formally.  
    7) If someone greets or give feedback then give the response in positive and formalway!
    8)If the user asks an indirect question (e.g., asking about "Access and Obselete" without specific details), perform a broader search in the vector database to identify the most relevant policy and its details.
            Do not limit the search to direct matches. Instead, interpret the question in the broader context of the user’s potential need and provide the most comprehensive, relevant answer from the database.
            Detect the user's underlying intent by analyzing the wording of the query and provide a detailed response, focusing on what the user might be seeking even if not explicitly stated.
            Example of indirect question: During sox reviw what is access and obselete. This can imply need of a excess and obselete related policy, Sox related policies.
    9)"When a user types a keyword, identify and process it accordingly:

        For 'VAM':
        Interpret this as 'Value Added Margin.' Retrieve and provide information related to Value Added Margin, including definitions, calculations, recent statistics, and examples within the database.

        For 'PSL':
        Recognize this as 'Preferred Supplier List.'
         
        For 'CMT':  
        Recognize this as 'Claim Management Tracking.

        For 'CMF':  
        Recognize this as 'Customer Master File'.
        Look for data concerning the Preferred Supplier List,  within the database.
    
    10) Ensure that at the end of each answer, you provide the policy number, policy name, and policy owner related to the answer using the correct document. Extract this information directly 
    from the document and present it in dictionary format within curly brackets. Do not generate or guess the policy details


{context}
"""

    # Access the persistent vector store
    vector_store = vector_stores[collection_name]

    # Create the PromptTemplate object
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", base_prompt), ("human", "{input}")]
    )

    # Create a retriever for the vector store
    retriever = vector_store.as_retriever()

    # Create a document chain to combine retrieved context
    combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)

    # Create the retrieval chain
    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain,
    )

    return retrieval_chain


def search_collections(chatbot_type, query):
    # Loop through each collection
    for collection_name in collection_names:
        try:
            # Create a retrieval chain for the current collection
            retrieval_chain = create_state_independent_retrieval_chain(collection_name, query)

            # Query the retrieval chain
            response = retrieval_chain.invoke({"input": query})

            # Check if the answer exists and contains specific "not found" phrases
            answer = response.get("answer", "")

            # If answer contains "sorry" or "not found", move to the next collection
            if any(keyword in answer.lower() for keyword in ["sorry", "not found", "does not exist"]):
                continue  # Skip to the next collection if the answer is not valid

            return answer

        except Exception as e:
            return jsonify("Error processing collection. Please contact the support team.")
            continue  # Continue to the next collection if there's an error

    # If no answer is found in any collection
    return "I am sorry, the answer may not be available in the database. Please try rephrasing your query."

# Define the list of collection names and create persistent vector stores
IT_collection_names = [
    "IT_Batch1",
    "IT_Batch2",
    "IT_Batch3",
    "IT_Batch4", 
]

# NOTE: point this at a local/relative path (or an env var) rather than a
# machine-specific absolute path.
IT_DB_PATH = os.getenv("IT_DB_PATH", "./data/it_vector_db")

IT_vector_stores = {
    IT_collection_name: Chroma(
        collection_name=IT_collection_name,
        persist_directory=IT_DB_PATH,
        embedding_function=embeddings,
    )
    for IT_collection_name in IT_collection_names
}

def RAG(query):
    """
    Loop through all collections, retrieve relevant documents (vectors) for the answer,
    aggregate their content into one context, and then generate the answer via the LLM.
    """
    date = "what is the recent effective date of the " + query

    contexts = []
    for IT_collection_name in IT_collection_names:
        try:
            IT_vector_store = IT_vector_stores[IT_collection_name]
            IT_retriever = IT_vector_store.as_retriever()
            docs = IT_retriever.get_relevant_documents(query)
            docs2 = IT_retriever.get_relevant_documents(date)
            for doc in docs:
                contexts.append(doc.page_content)
            for doc in docs2:
                contexts.append(doc.page_content)
        except Exception as e:
            jsonify(f"Error retrieving from {IT_collection_name}")
            continue

    aggregated_context = "\n\n".join(contexts)

    answer_prompt = f"""
        Role: You are an IT auditor with over 20 years of experience. Your responsibility is to provide accurate and comprehensive answers strictly based on the IT policies vector database.
        If user greets you by saying Hi! or Hello or semantic to it greet them back and don't go in vector database.

        Instructions:

        Identify Relevant Policies:

        Analyze the user's query and extract relevant policies from the IT policies vector database.
        If the query is broad (e.g., "Application security"), focus primarily on the dedicated Application Security Policy. Supplement your answer with context from additional policies only if necessary.

        Reference Details for Each Policy:

        For each policy referenced in your answer, include its key details in a clearly structured "Policy References" section at the very beginning. Use the following format for each policy:

        Document number: <document number>
        Document name: <document name>
        Effective Date: <search for the recent effective date in the given context>
        Approver's name: <approver name>
        Owner's name: <owner name>

        If multiple policies are used, list each one separately under its own heading (e.g., Policy 1, Policy 2, etc.) to ensure clarity on which parts of the answer are derived from which source.

        Answer Structure:

        Policy References: Start your answer with a summary section listing all the relevant policies and their key details as described above.
        Comprehensive Response: Follow with a detailed answer addressing the user's query. In the answer, clearly indicate which policy each piece of information comes from, ensuring full traceability to the source.

        Focus and Source:

        Your response must rely solely on information from the IT policies vector database.
        Focus on the user's underlying intent and provide actionable, precise guidance based on the referenced policies.

        Context:

        {aggregated_context}

        Query:

        {query}

        Answer:
        """
    answer_response = llm.invoke(answer_prompt)
    return str(answer_response.content)

def RAG_question(query):
    """
    Loop through all collections, retrieve relevant documents for follow-up questions,
    aggregate their content into one context, and then generate four follow-up/suggestive questions via the LLM.
    """
    contexts = []
    for IT_collection_name in IT_collection_names:
        try:
            IT_vector_store = IT_vector_stores[IT_collection_name]
            IT_retriever = IT_vector_store.as_retriever()
            docs = IT_retriever.get_relevant_documents(query)
            for doc in docs:
                contexts.append(doc.page_content)
        except Exception as e:
            jsonify(f"Error retrieving from {IT_collection_name}")
            continue

    aggregated_context = "\n\n".join(contexts)

    question_prompt = f"""
    Role: You are an IT auditor with over 20 years of experience. Based strictly on the IT policies vector database, generate four follow-up or suggestive questions that align with the user's underlying intent.

    Your Tasks:
    1. Broaden the context or perspective of the topic.
    2. Clarify potential ambiguities within the topic area.
    3. Encourage consideration of related IT policy implications or best practices.
    4. Avoid direct references to the user such as the words "you" or "I".
    Context:
    {aggregated_context}

    Query:
    {query}

    Follow-up Questions:
    """
    question_response = llm.invoke(question_prompt)
    return str(question_response.content)


def IT_search_collections(query):
    """
    Use separate functions to retrieve the answer and follow-up questions,
    then combine their outputs into one final response.
    """
    answer = RAG(query)

    # Regex pattern for greeting detection
    greeting_pattern = r"\b(hi|hello|hey|good\s+(morning|evening|afternoon))\b"

    # Check if the input is a greeting (case-insensitive)
    if re.search(greeting_pattern, query, re.IGNORECASE):
        return answer

    follow_up_questions = RAG_question(query)

    final_answer = (
        answer +
        "\n\n**Based on the answer, the following follow-up questions are suggested:**\n\n" +
        follow_up_questions
    )
    return final_answer

###*************ERM starts here **********************************************
# NOTE: point this at a local/relative path (or an env var) rather than a
# machine-specific absolute path.
db_file_path = os.getenv("RISK_DB_PATH", "./data/risk_data.db")
engine = create_engine(f"sqlite:///{db_file_path}")

db = SQLDatabase(engine)

db_chain = SQLDatabaseChain(llm=llm_agent, database=db, verbose=True, return_intermediate_steps=True, use_query_checker=True)

def retrieve_from_db(query: str) -> str:
    db_context = db_chain(query)
    sql_result = db_context['intermediate_steps'][3]
    sql_script = db_context['intermediate_steps'][1]
    db_context = db_context['result'].strip()
    return db_context, sql_result, sql_script

prompt_columns = [
    "Type", "Region", "Country", "Segment", "Entity", "Category", "Risk Name", "Process Risk Category",
    "Functional Risk Category", "Enterprise Risk Category", "Risk Description", "Risk Causes", "Risk Impacts",
    "Risk Response Actions", "Risk Owner", "Risk Status", "Risk Mitigation", "L", "Likelihood", "I", "Impact", "Risk Rating", "Risk_level"
]

column_list = ", ".join(prompt_columns)

def generate(query: str) -> str:
    prompt = (
        f"As an expert in data analysis on risk registers data, please provide a detailed answer for the user query '{query}'. ""\n"
        f"Ensure to base your answers on columns such as {column_list}. " "\n"
        f"Provide a short note on how you have come up with the answer without stating anything about the SQL background so that the user understand on how the answer is retrieved"
        f"**Provide the response as an insightful readable summary or answer. Make sure you explain the user on how the answer is given so that they understand how the answer is searched in the database.**"        
        
        f"Condition 1: Do not use 'LIMIT', 'MAX', 'MIN' statement in the query, Unless the query absolutely requires it." "\n"
        f"Condition 2: Ensure that you do not give SQL script as answer " "\n"
        f"Condition 3: Ensure that you do not use columns which are not required by the query. for example, we need not use Business_Type in WHERE clause unless required by the query" "\n" 
        f"Condition 4: Ensure that you take only the necessary columns to query and filter the database" "\n"
    )

    db_context, sql_result, sql_script = retrieve_from_db(prompt)

    messages = [
    SystemMessage(content="""If the answer requires a structured format to make it readable, provide the final answer like the below format:
        **Site**
        **Risk Category**
        **Process Risk Category**
        **Functional Risk Category**
        **Enterprise Risk Category**
        **Region and Country Details**
        **Risk Description**
        **Impact Description**
        **Cause of the Risk**
        **Mitigation status**
        **Risk Response Actions**
        **Risk Level**
        **Date**

        Else, provide the response as an insightful, readable summary or answer without the above format. Ensure you don't give SQL scripts or Python code as an answer.

        Important: Do not include Risk ID, likelihood, impact, and risk rating in the final response"""),
    HumanMessage(content=f"User Query: {query}\n\nContext: {db_context}")
    ]

    response = llm(messages).content
    return response, sql_result, sql_script
    # return db_context

# Dictionary to store collection names for each risk level
ERM_collection_names = {
    "All": [f"chroma_All_part{i+1}" for i in range(11)],
    "High": [f"chroma_High_part{i+1}" for i in range(2)],
    "Medium": [f"chroma_Medium_part{i+1}" for i in range(4)],
    "Low": [f"chroma_Low_part{i+1}" for i in range(4)],
    "Immaterial": [f"chroma_Immaterial_part{i+1}" for i in range(2)]
}

def get_chroma_db(risk_level: str):
    allowed_levels = {"All", "High", "Medium", "Low", "Immaterial"}
    if risk_level not in allowed_levels:
        raise ValueError("Invalid risk level")

    persist_dir = os.path.abspath(os.path.join("chroma_data_prod", risk_level))
    collections = ERM_collection_names.get(risk_level, [])

    chroma_collections = []
    for collection in collections:
        print(f"Loading collection {collection} from {persist_dir}")
        chroma = Chroma(
            persist_directory=persist_dir,
            collection_name=collection,
            embedding_function=embeddings
        )
        data = chroma.get()
        if data and len(data['documents']) > 0:
            chroma_collections.append(chroma)
            print(f"Collection {collection} loaded with {len(data['documents'])} documents.")
        else:
            print(f"Collection {collection} is empty.")

    print(f"Total collections loaded: {len(chroma_collections)}")
    return chroma_collections


def get_ensemble_retriever(chroma_collections, search_kwargs):
    if not chroma_collections:
        print("No collections to create retrievers from.")
        return None

    retrievers = [chroma.as_retriever(search_kwargs=search_kwargs) for chroma in chroma_collections]
    ensemble_retriever = EnsembleRetriever(retrievers=retrievers, weights=[1.0] * len(retrievers))
    return ensemble_retriever

def get_retriever(query, k=99):
    high_risk_keywords = ["top", "major", "high", "critical", "severe", "significant", "urgent"]
    medium_risk_keywords = ["moderate", "medium", "balanced", "average", "manageable", "acceptable"]
    low_risk_keywords = ["low", "minor", "negligible", "small", "minimal", "reduced", "trivial"]
    immaterial_risk_keywords = ["immaterial", "insignificant", "inconsequential", "irrelevant", "petty", "unimportant"]
    top_10 = ["top", "major", "critical"]

    if any(keyword in query.lower() for keyword in high_risk_keywords):
        chroma_collections = get_chroma_db("High")
        search_kwargs = {"k": k, "filter": {"Risk level": "high"}}
        if any(keyword in query.lower() for keyword in top_10):
            search_kwargs["filter"] = {'$and': [{'priority': {'$eq': "top"}}, {'Risk level': {'$eq': "high"}}]}
    elif any(keyword in query.lower() for keyword in medium_risk_keywords):
        chroma_collections = get_chroma_db("Medium")
        search_kwargs = {"k": k, "filter": {"Risk level": "medium"}}
        if any(keyword in query.lower() for keyword in top_10):
            search_kwargs["filter"] = {'$and': [{'priority': {'$eq': "top"}}, {'Risk level': {'$eq': "medium"}}]}
    elif any(keyword in query.lower() for keyword in low_risk_keywords):
        chroma_collections = get_chroma_db("Low")
        search_kwargs = {"k": k, "filter": {"Risk level": "low"}}
        if any(keyword in query.lower() for keyword in top_10):
            search_kwargs["filter"] = {'$and': [{'priority': {'$eq': "top"}}, {'Risk level': {'$eq': "low"}}]}
    elif any(keyword in query.lower() for keyword in immaterial_risk_keywords):
        chroma_collections = get_chroma_db("Immaterial")
        search_kwargs = {"k": k, "filter": {"Risk level": "immaterial"}}
        if any(keyword in query.lower() for keyword in top_10):
            search_kwargs["filter"] = {'$and': [{'priority': {'$eq': "top"}}, {'Risk level': {'$eq': "immaterial"}}]}
    else:
        chroma_collections = get_chroma_db("All")
        search_kwargs = {"k": 30}
        if any(keyword in query.lower() for keyword in top_10):
            search_kwargs["filter"] = {"priority": "top"}
    return get_ensemble_retriever(chroma_collections, search_kwargs)

def ERM_RAG(query):
    base_prompts = """
        Role:
        You are an Enterprise Risk Management expert assistant responding **strictly and exclusively** based on the ERM risk register vector database. All your answers must be 100% grounded in the retrieved data only. You may not synthesize, infer, summarize, or assume beyond the data retrieved. If a detail (including risk level) does not explicitly exist in the context, you must not include it in your answer.

        General Instructions:

        - Review all retrieved database entries thoroughly before answering.
        - Only use information that is explicitly included in the context provided.
        - Never reference external sources or prior knowledge.
        - Do not invent or extrapolate any content, including severity, region details, or risk implications.
        - Always maintain a formal tone and include a greeting only when the user greets you.

        Important Enforcement Rules:

        - You may only state a **Risk level** (High, Medium, Low, or Immaterial) **if it is explicitly labeled as such in the context data**.
            - If **no row has a risk labeled "High"**, then strictly **do not state** or imply "High Risk" under any circumstance.
            - **Do not summarize** multiple medium risks into a high risk.
        - If a category, theme, country or risk level is not present in the retrieved data, you must say so.
        - Do not hallucinate missing parts of the structure—report them as not available if needed.

        Answer Structure:

        A. For General/Broad Questions:

        Provide details for **10 distinct risk categories**, each with the following fields (when available in the context):

        - **Category Name**: (from `Category`)
        - **Process Risk Category**: (from  `Enterprise Risk Category`)
        - **Functional Risk Category**: (from `Functional Risk Category`)
        - **Enterprise Risk Category**: (from  `Process Risk Category`)
        - **Risk Name**: (from `Risk Name`)
        - **Region and Country Details**: (from `Region`, `Country`, `Entity`)
        - **Risk Description**: (from `Risk Description`)
        - **Impact Description**: (from `Risk Impacts`)
        - **Cause of the Risk**: (from `Risk Causes`)
        - **Risk Response Actions**: (from `Risk Response Actions`)
        - **Risk Status Details**: (from `Risk Status`, `Risk Mitigation`)
        - **Risk Owner**: (from `Risk Owner`)
        - **Risk Level**: Use the risk level **exactly as labeled** in the data (High, Medium, Low, Immaterial). If missing, say “Not specified”.

        Risk Status details must not be summarized from several IDs which have different created, modified, created by or modified by data.

        **Strictly DO NOT include or display: Risk ID, Likelihood, Impact, or internal risk rating calculations. These are not needed in the Final Answer for the user**

        B. For Targeted/Specific Questions:

        - Focus only on requested locations, sites, themes, or categories.
        - Use a side-by-side comparison or structured bullets to present differences.
        - If data is not available for part of the query (e.g., no data for a given country), state that clearly and do not assume or fabricate details.

        C. Analytical or Comparative Questions:

        - Begin by identifying the relevant risk categories from the context.
        - Provide segmented insights using only the data.
        - Do not include trends or insights that are not explicitly stated in the context.

        Final Reminder:

        Your answer must be '100%' data-grounded from the ERM risk register vector database. Do not include any content that is not explicitly found in the context.
        Even If there is only one or few context available to proceed, continue to give the answer only with the given context.

        Context:
        {context}

        Question:
        {question}

        Answer:
    """

    retriever = get_retriever(query, k=99)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    prompt_template = PromptTemplate(template=base_prompts, input_variables=["context"])

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ERM_llm,
        memory=memory,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

    response = qa_chain.run(query)

    return response

# re.search(r'\bLIKE\b', sql_script, re.IGNORECASE) or 

def contains_theme_or_themes(text):
    # Check if 'theme' or 'themes' exists as a word
    return bool(re.search(r'\btheme(s)?\b', text, re.IGNORECASE))

def answer(query: str) -> str:
    try:
        # agents_answer, sql_result, sql_script = generate(query, choice)
        agents_answer, sql_result, sql_script = generate(query)
        if re.search(r'\b(WHERE|GROUP BY|ORDER BY)\b.*\b(Risk_Description|Risk_Causes|Risk_Impacts)\b',
                   sql_script, re.IGNORECASE):
            return ERM_RAG(query)
        # check_script = check_sql_condition(sql_script, query, choice)
        # Check if the answer is None, too short, or contains an SQL script
        elif len(sql_result) == 0 or len(agents_answer) < 3 or contains_theme_or_themes(query):
            return ERM_RAG(query)
        # If the agents' answer is valid, return it
        else:
            return agents_answer
    except openai.BadRequestError as e:
        # if "Request body size exceeds the configured limit" in str(e):
            return ERM_RAG(query)



csp = {
    'default-src': ["'none'"],
    'connect-src': ["'self'"],
    'object-src': ["'none'"],
    'base-uri': ["'self'"],
    'block-all-mixed-content': '',
    'upgrade-insecure-requests': '',
}
# strict transport security to avoid man-in-the-middle attacks, and force HTTPS only
# talisman= Talisman(app,content_security_policy=csp,strict_transport_security=True,strict_transport_security_max_age=31536000,force_https=True)
security_headers = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'Referrer-Policy': 'no-referrer',
}

Talisman(
    app,
    content_security_policy=csp,
    strict_transport_security=True,
    strict_transport_security_max_age=31536000,
    force_https=True,
    frame_options='DENY',
    content_security_policy_nonce_in=['script-src'],
    session_cookie_secure=True,
    session_cookie_http_only=True,
    session_cookie_samesite='Strict',
    referrer_policy='no-referrer',
)


# Whitelisted domains (trusted) — replace with your own Okta/IdP domain via env var
TRUSTED_OKTA_DOMAINS = [os.getenv("OKTA_TRUSTED_DOMAIN", "https://your-company.okta.com")]

def okta_introspect(token, client_id, okta_domain):
    # NOTE: the Okta authorization server ID below is a placeholder — replace
    # with your own org's authorization server ID (set via env var ideally).
    okta_auth_server_id = os.getenv("OKTA_AUTH_SERVER_ID", "default")
    okta_introspect = f"{okta_domain}/oauth2/{okta_auth_server_id}/v1/introspect"
    payload = {
        'client_id': client_id,
        'token': token,
        'token_type_hint': 'access_token'
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    response = requests.post(okta_introspect, data=payload, headers=headers)

    return response.json()

def okta_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Extract the Authorization header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization'].split()
            if len(auth_header) == 2 and auth_header[0] == "Bearer":
                token = auth_header[1]
            else:
                return jsonify({"error": "Invalid Authorization header format"}), 401
        else:
            return jsonify({"response": "Authorization header is missing!"}), 401

        if not token:
            return jsonify({"response": "Token is missing!"}), 401

        client_id = os.getenv("OKTA_CLIENT_ID")
        okta_domain = os.getenv("OKTA_DOMAIN")

        # Validate environment variables
        if not client_id or not okta_domain:
            return jsonify({"response": "Server configuration error: missing Okta credentials"}), 500

        # Introspect the token with Okta
        introspect_response = okta_introspect(token, client_id, okta_domain)

        # # Check if the token is active
        # if introspect_response.get('active'):
        #     return f(*args, **kwargs)
        # else:
        #     return jsonify({"response": "Invalid or inactive token"}), 401
        if introspect_response.get('active'):
            # Store user info in Flask global context
            g.okta_user = introspect_response.get('username')
            g.user_region = introspect_response.get("region")
            g.user_country = introspect_response.get("country")
            return f(*args, **kwargs)
        else:
            return jsonify({"response": "Invalid or inactive token"}), 401

    return decorated_function



def clean_answer(answer):
    # Remove ** patterns
    cleaned_answer = re.sub(r'\*\*', '', answer)
    cleaned_answer = re.sub(r'###', '', cleaned_answer)  # Remove ###
    cleaned_answer = re.sub(r'-', '', cleaned_answer)

    return cleaned_answer

def dev_okta_introspect(token, dev_client_id, dev_okta_domain):
    dev_auth_server_id = os.getenv("DEV_OKTA_AUTH_SERVER_ID", "default")
    okta_introspect = f"{dev_okta_domain}/oauth2/{dev_auth_server_id}/v1/introspect"
    payload = {
        'client_id': dev_client_id,
        'token': token,
        'token_type_hint': 'access_token'
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    response = requests.post(okta_introspect, data=payload, headers=headers)

    return response.json()

def dev_okta_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            # Extract the Authorization header
            auth_header = request.headers.get('Authorization', '')
            parts = auth_header.split()

            if len(parts) != 2 or parts[0] != "Bearer":
                return jsonify({
                    "response": "Invalid or missing authorization format. Please login again.",
                    "error": True,
                    "status_code": 401
                }), 200

            token = parts[1]

            if not token:
                return jsonify({
                    "response": "Token is missing. Please login again.",
                    "error": True,
                    "status_code": 401
                }), 200

            dev_client_id = os.getenv("DEV_OKTA_CLIENT_ID")
            dev_okta_domain = os.getenv("DEV_OKTA_DOMAIN")

            if not dev_client_id or not dev_okta_domain:
                return jsonify({
                    "response": "Server error: Okta credentials are not configured properly.",
                    "error": True,
                    "status_code": 500
                }), 200

            # Introspect token
            introspect_response = dev_okta_introspect(token, dev_client_id, dev_okta_domain)

            if introspect_response.get('active'):
                return f(*args, **kwargs)
            else:
                return jsonify({
                    "response": "Your session has expired or the token is invalid. Please log in again.",
                    "error": True,
                    "status_code": 401
                }), 200

        except Exception as e:
            # Log actual error internally
            print("Unexpected Okta validation error:", e)
            return jsonify({
                "response": "Unexpected error occurred during authentication. Please try again later.",
                "error": True,
                "status_code": 500
            }), 200

    return decorated_function

@app.route("/get", methods=["POST"])
@okta_required
def chat():
    data = request.get_json()
    chatbot_type = (data.get("chatbot", "")).lower()
    if not chatbot_type:
        return jsonify({"response": "Chatbot type is required."}), 400

    query = escape(data.get("msg"))
    if not query:
        return jsonify({"response": "Query message is required."}), 400
    oracle_connection = get_oracle_connection()
    try:
        if chatbot_type == "finance":
            try:
                answer = search_collections(chatbot_type, query)
                cleaned_answer = clean_answer(answer)
                # return jsonify({"response": (cleaned_answer)})
            except Exception as e:
                return jsonify({"response": "Please contact the support team."}), 500

        elif chatbot_type == "it":
            try:
                answer = IT_search_collections(query)
                cleaned_answer = clean_answer(answer)
                # return jsonify({"response": (cleaned_answer)})
            except Exception as e:
                return jsonify({"response": "Please contact the support team."}), 500

        else:
            return jsonify({"response": "The current chatbot development is actively underway."})


        cleaned_answer = clean_answer(answer)

            # === LOGGING TO ORACLE DATABASE ===
            # Static values (for now)
        user_email = g.okta_user
        site_code = None
        country = g.user_country
        region = g.user_region
        timestamp = datetime.now()
        chat_number = None
        chatbot_id = 1 if chatbot_type == "finance" else 2
        # Insert log into Oracle DB
        cursor = oracle_connection.cursor()
        insert_sql = """
                    INSERT INTO CHATBOT_LOG
                        (QUESTION_VAR, QNA_ANSWER_VAR, CURRENT_USER, LOGIN_DATE_TIME, CHAT_BOT_ID, 
                        SITECODE, USER_COUNTRY, USER_REGION, CHATNUMBER) 
                    VALUES 
                        (:1, :2, :3, :4, :5, :6, :7, :8, :9)
                """
        cursor.execute(insert_sql, (
                    query,
                    cleaned_answer,
                    user_email,
                    timestamp,
                    chatbot_id,
                    site_code,
                    country,
                    region,
                    chat_number
                ))
        oracle_connection.commit()
        cursor.close()

            # === RETURN RESPONSE ===
        return jsonify({"response": cleaned_answer})

    except Exception as e:
        return jsonify({"response": "Please contact the support team."}), 500


@app.route('/geterm', methods=['POST'])
@dev_okta_required
def ask():
    data = request.get_json()

    # Validate input
    if not data:
        return jsonify({'response': "Invalid input."}), 400

    # Extract 'msg' field
    raw_msg = data.get('msg', '').strip()
    if not raw_msg:
        return jsonify({'response': "Message is missing."}), 400

    query = escape(str(raw_msg))
    chatbot_type = "ERM Risk Register"

    # Call your answer generation logic
    final_answer = answer(query)

    if final_answer is None:
        return jsonify({'response': "Sorry, I couldn't generate an answer at this time."})

    disclaimer = (
        "Disclaimer: The responses provided by this bot are generated by an AI model "
        "and may not be completely accurate. Please exercise caution and verify the "
        "information with reliable sources for critical decisions."
    )

    final_answer_with_disclaimer = final_answer + "\n\n" + disclaimer
    cleaned_answer_erm = clean_answer(final_answer_with_disclaimer)
    return jsonify({'response': cleaned_answer_erm})

if __name__ == '__main__':
    # app.run(host='0.0.0.0',port='5000',debug=False,ssl_context=('server_cert.cer','server_key.key'))
    app.run(host='0.0.0.0', port='80', debug=False)
