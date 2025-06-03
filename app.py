import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, UnstructuredImageLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
import os
import tempfile
import json
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import random
from export_utils import (
    export_to_pdf, 
    export_to_excel, 
    export_to_json, 
    get_download_link,
    create_sample_visualization
)

# Load environment variables
load_dotenv()

# Init OpenAI and embedding
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
client = OpenAI(base_url="https://models.inference.ai.azure.com", api_key=os.environ["GITHUB_TOKEN"])

# Helper: Get company-specific vector DB path
def get_vector_path(company_name):
    return f"faiss_{company_name.replace(' ', '_').lower()}_index"

# Helper: Load and split documents
def load_and_split(file):
    if file.type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            loader = PyPDFLoader(tmp.name)
    elif file.type == "text/csv":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(file.read())
            loader = CSVLoader(file_path=tmp.name)
    elif file.type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        # For Excel files (.xls or .xlsx)
        suffix = ".xlsx" if file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" else ".xls"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.read())
            
            try:
                # Try to determine the engine based on file extension
                if suffix == ".xlsx":
                    engine = "openpyxl"  # For .xlsx files
                else:
                    engine = "xlrd"      # For .xls files
                
                # Convert Excel to CSV for processing
                excel_df = pd.read_excel(tmp.name, engine=engine)
                csv_path = tmp.name.replace(suffix, '.csv')
                excel_df.to_csv(csv_path, index=False)
                loader = CSVLoader(file_path=csv_path)
            except Exception as e:
                st.error(f"Error processing Excel file: {str(e)}")
                st.info("Make sure you have the required packages installed: pip install openpyxl xlrd")
                return []
    elif file.type.startswith("image"):
        image = Image.open(file)
        image.save("temp_image.png")
        loader = UnstructuredImageLoader("temp_image.png")
    else:
        st.warning(f"Unsupported file type: {file.type}")
        return []

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

# Helper: Generate response using LLM
def generate_response(user_input, vectordb):
    retriever = vectordb.as_retriever(score_threshold=0.7)
    docs = retriever.invoke(user_input)
    context = "\n".join([doc.page_content for doc in docs]) if docs else "No relevant documents found."

    prompt = f"""
    You are TelcoBot, an AI analyst for telecom companies.
    Use the context below to answer the user's question.
    If the question asks for analytics or statistics, try to provide numerical insights.
    If the context doesn't contain relevant information, acknowledge that and provide general telecom industry knowledge.
    Do not provide information that are not related to telecom industry, like politics, sports, personal life etc.
    Always maintain conversations in the telecom industry and try to work more within context
    
    Context:
    {context}

    Question:
    {user_input}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content, context

# Helper: Extract structured data from response
def extract_structured_data(response_text, query_type):
    """
    Extract structured data from the response text based on query type.
    This is a simplified version for the demo - in production, you would use more robust parsing.
    """
    # For demo purposes, generate sample data
    if "performance" in query_type.lower() or "uptime" in query_type.lower():
        # Sample network performance data
        sites = ['Site A', 'Site B', 'Site C', 'Site D', 'Site E']
        uptime = [99.8, 97.5, 99.9, 98.2, 99.5]
        latency = [15, 22, 12, 18, 14]
        
        return pd.DataFrame({
            'Site': sites,
            'Uptime (%)': uptime,
            'Latency (ms)': latency
        })
    
    elif "complaint" in query_type.lower() or "issue" in query_type.lower():
        # Sample customer complaint data
        categories = ['Network Outage', 'Slow Speed', 'Billing Issues', 
                      'Customer Service', 'Equipment Failure']
        counts = [45, 78, 32, 28, 15]
        
        return pd.DataFrame({
            'Category': categories,
            'Count': counts
        })
    
    elif "traffic" in query_type.lower() or "usage" in query_type.lower():
        # Sample site traffic data
        hours = list(range(24))
        traffic = [20, 15, 10, 8, 5, 10, 25, 45, 60, 70, 75, 80, 
                  85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 30, 25]
        
        return pd.DataFrame({
            'Hour': hours,
            'Traffic (Gbps)': traffic
        })
    
    # Default case - return None if no structured data can be extracted
    return None

# Helper: Generate sample telecom queries
def get_sample_queries():
    return [
        "What are the top 5 issues reported at Site A this month?",
        "Compare network uptime between Site B and Site C for the last week.",
        "Generate a performance report for the Eastern region sites.",
        "What was the average latency across all sites yesterday?",
        "Show me the customer complaint trends for the last quarter.",
        "What maintenance activities are scheduled for next week?",
        "Analyze the traffic patterns for Site D during peak hours.",
        "Which sites had equipment failures in the past 30 days?",
        "Summarize the network upgrade impact on customer satisfaction."
    ]

# Add custom CSS
def local_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #0053A0;
    }
    .stButton>button {
        background-color: #0053A0;
        color: white;
    }
    .download-button {
        display: inline-block;
        padding: 8px 16px;
        background-color: #0053A0;
        color: white;
        text-decoration: none;
        border-radius: 4px;
        margin: 5px;
    }
    .download-button:hover {
        background-color: #003b73;
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #e6f3ff;
        border-left: 5px solid #0053A0;
    }
    .bot-message {
        background-color: #f0f0f0;
        border-left: 5px solid #666;
    }
    .sidebar .sidebar-content {
        background-color: #0053A0;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit UI
st.set_page_config(page_title="TelcoRAG Demo", layout="wide")
local_css()

# App header
st.title("📡 Telecom AI Report Assistant")
st.markdown("### Intelligent Analytics & Reporting for Telecom Operations")

# Login Section
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.subheader("🔐 Company Login")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image("https://img.freepik.com/free-vector/telecommunication-isometric-composition-with-images-communication-towers-satellite-dishes-with-people-workers-vector-illustration_1284-30375.jpg", width=300)
    
    with col2:
        with st.form("login_form"):
            company_name = st.text_input("Company Name")
            access_key = st.text_input("Access Key", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if access_key == "1234":  # Demo password
                    st.session_state.authenticated = True
                    st.session_state.company_name = company_name
                    st.success(f"Welcome, {company_name}!")
                else:
                    st.error("Invalid access key")
    st.stop()

# Company-specific context
company_name = st.session_state.company_name
VECTOR_DIR = get_vector_path(company_name)
if os.path.exists(VECTOR_DIR):
    vectordb = FAISS.load_local(VECTOR_DIR, embedding, allow_dangerous_deserialization=True)
else:
    vectordb = None

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "report_history" not in st.session_state:
    # For demo, create some sample reports
    st.session_state.report_history = [
        {
            "title": "Network Performance Analysis",
            "date": datetime.datetime.now() - datetime.timedelta(days=7),
            "type": "PDF",
            "query": "Analyze network performance for all sites"
        },
        {
            "title": "Customer Complaint Summary",
            "date": datetime.datetime.now() - datetime.timedelta(days=14),
            "type": "Excel",
            "query": "Summarize customer complaints by category"
        },
        {
            "title": "Site Maintenance Schedule",
            "date": datetime.datetime.now() - datetime.timedelta(days=21),
            "type": "JSON",
            "query": "Generate maintenance schedule for Q3"
        }
    ]

# Sidebar navigation
with st.sidebar:
    st.header(f"Welcome {company_name}")
    st.image("https://img.freepik.com/free-vector/isometric-telecommunications-industry-composition-with-isolated-image-cellular-communication-tower-with-satellite-dishes-vector-illustration_1284-30376.jpg", width=200)
    
    page = st.radio("Navigate", [
        "📊 Dashboard", 
        "📁 Upload Documents", 
        "💬 Chat with Data",
        "📈 Analytics",
        "📚 Report History"
    ])
    
    st.markdown("---")
    st.markdown("### Sample Telecom Queries")
    
    # Display sample queries that can be clicked
    for query in get_sample_queries():
        if st.button(query, key=f"sample_{query}"):
            st.session_state.selected_query = query
            st.session_state.page = "💬 Chat with Data"
            # Force a rerun to switch to the chat page
            # st.experimental_rerun()
            st.rerun()

if page == "📊 Dashboard":
    st.subheader("📌 Telecom Operations Dashboard")
    
    # Create a dashboard layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Network Performance")
        # Create a sample visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        sites = ['Site A', 'Site B', 'Site C', 'Site D', 'Site E']
        uptime = [99.8, 97.5, 99.9, 98.2, 99.5]
        ax.bar(sites, uptime, color='#0053A0')
        ax.set_ylim([95, 100])
        ax.set_ylabel('Uptime (%)')
        ax.set_title('Site Uptime - Last 7 Days')
        st.pyplot(fig)
        
        st.markdown("### Recent Alerts")
        alerts_df = pd.DataFrame({
            'Time': [
                '2023-07-01 08:23', 
                '2023-07-01 12:45', 
                '2023-07-02 03:12'
            ],
            'Site': ['Site B', 'Site D', 'Site A'],
            'Alert': [
                'Latency spike detected', 
                'Bandwidth utilization > 90%', 
                'Power supply failure'
            ],
            'Severity': ['Medium', 'High', 'Critical']
        })
        st.dataframe(alerts_df, use_container_width=True)
    
    with col2:
        st.markdown("### Customer Complaints")
        # Create a sample pie chart
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ['Network Outage', 'Slow Speed', 'Billing Issues', 
                      'Customer Service', 'Equipment Failure']
        counts = [45, 78, 32, 28, 15]
        ax.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90, 
               colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
        ax.axis('equal')
        st.pyplot(fig)
        
        st.markdown("### System Status")
        status_df = pd.DataFrame({
            'System': [
                'Core Network', 
                'Billing System', 
                'Customer Portal', 
                'Field Service App',
                'Monitoring System'
            ],
            'Status': ['Online', 'Online', 'Online', 'Degraded', 'Online'],
            'Uptime': ['99.99%', '99.95%', '99.9%', '98.5%', '99.99%']
        })
        st.dataframe(status_df, use_container_width=True)
    
    st.markdown("### Instructions & Overview")
    st.markdown("""
    Welcome to your Telecom AI workspace.

    **Instructions:**
    1. Navigate to **Upload Documents** to upload PDFs, CSVs, or images.
    2. Use **Chat with Data** to ask intelligent questions about your reports.
    3. Download AI-generated answers in multiple formats (PDF, Excel, JSON).
    4. View analytics visualizations based on your data.
    5. Access your report history to review previous analyses.

    **Note:** This demo showcases the capabilities of the Multimodal RAG system for telecom operations.
    """)

elif page == "📁 Upload Documents":
    st.subheader("📁 Upload Telecom Files")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader("Upload PDFs, CSVs, or Images", accept_multiple_files=True)

        if uploaded_files:
            all_chunks = []
            for file in uploaded_files:
                with st.spinner(f"Processing {file.name}..."):
                    chunks = load_and_split(file)
                    if chunks:
                        all_chunks.extend(chunks)
            if all_chunks:
                with st.spinner("Adding to knowledge base..."):
                    if vectordb:
                        vectordb.add_documents(all_chunks)
                    else:
                        vectordb = FAISS.from_documents(all_chunks, embedding)
                    vectordb.save_local(VECTOR_DIR)
                    st.success("Documents added to your knowledge base ✅")
    
    with col2:
        st.markdown("### Supported File Types")
        st.markdown("""
        - **PDF Reports** (.pdf)
        - **Excel Spreadsheets** (.csv, .xlsx)
        - **Images** (.jpg, .png)
        
        Coming soon:
        - Word Documents (.docx)
        - PowerPoint Presentations (.pptx)
        - Audio Recordings (.mp3, .wav)
        """)
        
        st.markdown("### Document Processing")
        st.markdown("""
        When you upload documents, the system:
        
        1. Extracts text content
        2. Splits into manageable chunks
        3. Creates vector embeddings
        4. Stores in your private knowledge base
        5. Makes content available for AI queries
        """)

elif page == "💬 Chat with Data":
    st.subheader("💬 Ask About Your Telecom Data")

    # Check if we have a selected query from the sidebar
    if hasattr(st.session_state, 'selected_query'):
        query_input = st.session_state.selected_query
        # Clear it after use
        del st.session_state.selected_query
    else:
        query_input = ""

    with st.form("chat_form"):
        user_query = st.text_input("Enter your question", value=query_input)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            generate_pdf = st.checkbox("Generate PDF Report")
        with col2:
            generate_excel = st.checkbox("Generate Excel Report")
        with col3:
            generate_json = st.checkbox("Generate JSON Report")
            
        include_visualization = st.checkbox("Include visualization (if applicable)")
        
        submitted = st.form_submit_button("Get Answer")

        if submitted and user_query:
            if vectordb is None:
                st.warning("Please upload documents first.")
            else:
                with st.spinner("Analyzing your data..."):
                    response, context = generate_response(user_query, vectordb)
                    st.session_state.chat_history.append((user_query, response))
                    
                    # Extract structured data if possible
                    structured_data = extract_structured_data(response, user_query)
                    
                    # Generate reports if requested
                    if generate_pdf:
                        pdf_path = export_to_pdf(
                            user_query, 
                            response, 
                            context, 
                            include_visualization, 
                            structured_data
                        )
                        st.session_state.last_pdf = pdf_path
                    
                    if generate_excel:
                        excel_path = export_to_excel(
                            user_query, 
                            response, 
                            context, 
                            structured_data
                        )
                        st.session_state.last_excel = excel_path
                    
                    if generate_json:
                        json_path = export_to_json(
                            user_query,
                            response,
                            context,
                            structured_data
                        )
                        st.session_state.last_json = json_path
                    
                    # Add to report history
                    if generate_pdf or generate_excel or generate_json:
                        report_type = "PDF" if generate_pdf else ("Excel" if generate_excel else "JSON")
                        st.session_state.report_history.append({
                            "title": user_query[:50] + "..." if len(user_query) > 50 else user_query,
                            "date": datetime.datetime.now(),
                            "type": report_type,
                            "query": user_query
                        })

    # Display the response
    if st.session_state.chat_history:
        st.subheader("Response")
        question, answer = st.session_state.chat_history[-1]
        
        st.markdown(f"<div class='chat-message user-message'><strong>You:</strong> {question}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-message bot-message'><strong>TelcoBot:</strong> {answer}</div>", unsafe_allow_html=True)
        
        # Display visualization if applicable
        if include_visualization and submitted:
            st.subheader("Visualization")
            data_type = "network_performance"
            if "complaint" in user_query.lower() or "issue" in user_query.lower():
                data_type = "customer_complaints"
            elif "traffic" in user_query.lower() or "usage" in user_query.lower():
                data_type = "site_traffic"
                
            viz_buffer = create_sample_visualization(data_type)
            st.image(viz_buffer)
        
        # Display download links
        if submitted:
            st.subheader("Download Reports")
            download_links = []
            
            if generate_pdf and hasattr(st.session_state, 'last_pdf'):
                download_links.append(get_download_link(st.session_state.last_pdf, "📄 Download PDF"))
            
            if generate_excel and hasattr(st.session_state, 'last_excel'):
                download_links.append(get_download_link(st.session_state.last_excel, "📊 Download Excel"))
            
            if generate_json and hasattr(st.session_state, 'last_json'):
                download_links.append(get_download_link(st.session_state.last_json, "🔍 Download JSON"))
            
            if download_links:
                st.markdown(" ".join(download_links), unsafe_allow_html=True)
    
    # Display chat history
    if len(st.session_state.chat_history) > 1:
        with st.expander("View Chat History"):
            for i, (q, a) in enumerate(st.session_state.chat_history[:-1]):
                st.markdown(f"<div class='chat-message user-message'><strong>You:</strong> {q}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-message bot-message'><strong>TelcoBot:</strong> {a}</div>", unsafe_allow_html=True)
                if i < len(st.session_state.chat_history) - 2:
                    st.markdown("---")

elif page == "📈 Analytics":
    st.subheader("📈 Advanced Analytics")
    
    # Create tabs for different analytics views
    tab1, tab2, tab3 = st.tabs(["Network Performance", "Customer Insights", "Site Analysis"])
    
    with tab1:
        st.markdown("### Network Performance Metrics")
        
        # Time period selector
        time_period = st.selectbox(
            "Select Time Period",
            ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last Quarter"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Network uptime chart
            st.markdown("#### Network Uptime by Region")
            
            # Sample data
            regions = ['North', 'South', 'East', 'West', 'Central']
            uptime_values = [99.92, 99.85, 99.78, 99.95, 99.89]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(regions, uptime_values, color='#0053A0')
            ax.set_ylim([99.5, 100])
            ax.set_ylabel('Uptime (%)')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}%',
                        ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
        
        with col2:
            # Latency chart
            st.markdown("#### Average Latency by Region")
            
            # Sample data
            latency_values = [12, 18, 15, 10, 14]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(regions, latency_values, color='#66b3ff')
            ax.set_ylabel('Latency (ms)')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height} ms',
                        ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
        
        # Traffic over time
        st.markdown("#### Network Traffic Over Time")
        
        # Sample time series data
        dates = pd.date_range(end=datetime.datetime.now(), periods=24, freq='H')
        traffic = [20, 15, 10, 8, 5, 10, 25, 45, 60, 70, 75, 80, 
                  85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 30, 25]
        
        traffic_df = pd.DataFrame({
            'Time': dates,
            'Traffic (Gbps)': traffic
        })
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(traffic_df['Time'], traffic_df['Traffic (Gbps)'], marker='o', linestyle='-', color='green')
        ax.fill_between(traffic_df['Time'], traffic_df['Traffic (Gbps)'], alpha=0.2, color='green')
        ax.set_xlabel('Time')
        ax.set_ylabel('Traffic Volume (Gbps)')
        ax.set_title('24-Hour Traffic Pattern')
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.autofmt_xdate()
        
        st.pyplot(fig)
    
    with tab2:
        st.markdown("### Customer Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Customer complaints pie chart
            st.markdown("#### Customer Complaint Distribution")
            
            # Sample data
            categories = ['Network Outage', 'Slow Speed', 'Billing Issues', 
                          'Customer Service', 'Equipment Failure']
            counts = [45, 78, 32, 28, 15]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90, 
                   colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
            ax.axis('equal')
            
            st.pyplot(fig)
        
        with col2:
            # Resolution time
            st.markdown("#### Average Resolution Time by Issue Type")
            
            # Sample data
            resolution_times = [24, 12, 48, 36, 72]  # hours
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(categories, resolution_times, color='#ff9999')
            ax.set_ylabel('Resolution Time (hours)')
            ax.set_xticklabels(categories, rotation=45, ha='right')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height}h',
                        ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Customer satisfaction trend
        st.markdown("#### Customer Satisfaction Trend")
        
        # Sample time series data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        satisfaction = [72, 74, 73, 75, 78, 82]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(months, satisfaction, marker='o', linestyle='-', color='#0053A0', linewidth=3)
        ax.set_ylim([70, 85])
        ax.set_xlabel('Month')
        ax.set_ylabel('Satisfaction Score')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels on points
        for i, score in enumerate(satisfaction):
            ax.text(i, score + 0.5, f'{score}%', ha='center', fontweight='bold')
        
        st.pyplot(fig)
    
    with tab3:
        st.markdown("### Site Analysis")
        
        # Site selector
        site = st.selectbox(
            "Select Site",
            ["Site A", "Site B", "Site C", "Site D", "Site E"]
        )
        
        # Create tabs for different site metrics
        site_tab1, site_tab2, site_tab3 = st.tabs(["Performance", "Equipment", "Maintenance"])
        
        with site_tab1:
            st.markdown(f"#### {site} Performance Metrics")
            
            # Sample performance data
            dates = pd.date_range(end=datetime.datetime.now(), periods=7, freq='D')
            uptime = [99.8, 99.7, 99.9, 99.8, 99.6, 99.9, 99.8]
            latency = [12, 14, 11, 13, 15, 12, 13]
            
            # Create DataFrame
            perf_df = pd.DataFrame({
                'Date': dates,
                'Uptime (%)': uptime,
                'Latency (ms)': latency
            })
            
            # Plot
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            color = '#0053A0'
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Uptime (%)', color=color)
            ax1.plot(perf_df['Date'], perf_df['Uptime (%)'], color=color, marker='o')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_ylim([99.5, 100])
            
            ax2 = ax1.twinx()
            color = 'red'
            ax2.set_ylabel('Latency (ms)', color=color)
            ax2.plot(perf_df['Date'], perf_df['Latency (ms)'], color=color, marker='s')
            ax2.tick_params(axis='y', labelcolor=color)
            
            fig.autofmt_xdate()
            fig.tight_layout()
            
            st.pyplot(fig)
            
            # Display data table
            st.dataframe(perf_df)
        
        with site_tab2:
            st.markdown(f"#### {site} Equipment Status")
            
            # Sample equipment data
            equipment_df = pd.DataFrame({
                'Equipment': ['Router', 'Switch', 'Antenna', 'Power Supply', 'Cooling System'],
                'Status': ['Online', 'Online', 'Online', 'Online', 'Warning'],
                'Last Maintenance': ['2023-06-15', '2023-05-22', '2023-06-30', '2023-04-10', '2023-06-01'],
                'Utilization (%)': [65, 48, 72, 55, 90]
            })
            
            # Display equipment table
            # Display equipment table
            st.dataframe(equipment_df, use_container_width=True)
            
            # Display utilization chart
            st.markdown("#### Equipment Utilization")
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(equipment_df['Equipment'], equipment_df['Utilization (%)'], color=['#0053A0', '#0053A0', '#0053A0', '#0053A0', '#ff9999'])
            ax.set_xlabel('Utilization (%)')
            ax.set_xlim([0, 100])
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width}%',
                        ha='left', va='center', fontweight='bold')
            
            st.pyplot(fig)
        
        with site_tab3:
            st.markdown(f"#### {site} Maintenance Schedule")
            
            # Sample maintenance data
            maintenance_df = pd.DataFrame({
                'Date': ['2023-07-15', '2023-07-22', '2023-08-05', '2023-08-18', '2023-09-01'],
                'Type': ['Routine Check', 'Software Update', 'Hardware Replacement', 'Antenna Alignment', 'Full System Audit'],
                'Duration (hours)': [2, 4, 8, 3, 12],
                'Status': ['Scheduled', 'Scheduled', 'Pending Approval', 'Scheduled', 'Planning']
            })
            
            # Display maintenance table
            st.dataframe(maintenance_df, use_container_width=True)
            
            # Display maintenance calendar
            st.markdown("#### Maintenance Calendar")
            
            # Convert dates to datetime
            maintenance_df['Date'] = pd.to_datetime(maintenance_df['Date'])
            
            # Create a date range for the calendar
            start_date = datetime.datetime.now().replace(day=1)
            end_date = start_date + datetime.timedelta(days=60)
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Create a calendar dataframe
            calendar_df = pd.DataFrame(index=range(6), columns=range(7))  # 6 weeks, 7 days
            
            # Fill the calendar with day numbers
            day_idx = 0
            for week in range(6):
                for weekday in range(7):
                    if day_idx < len(date_range):
                        calendar_df.iloc[week, weekday] = date_range[day_idx].day
                        day_idx += 1
            
            # Create a figure for the calendar
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('tight')
            ax.axis('off')
            
            # Create the table
            weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            table = ax.table(cellText=calendar_df.values, colLabels=weekdays, 
                            loc='center', cellLoc='center')
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2)
            
            # Highlight days with maintenance
            for i, date in enumerate(maintenance_df['Date']):
                for week in range(6):
                    for weekday in range(7):
                        cell_value = calendar_df.iloc[week, weekday]
                        if cell_value == date.day and date.month == date_range[0].month:
                            table[(week+1, weekday)].set_facecolor('#ff9999')
                            table[(week+1, weekday)].set_text_props(fontweight='bold')
            
            st.pyplot(fig)

elif page == "📚 Report History":
    st.subheader("📚 Report History")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        report_type_filter = st.multiselect(
            "Filter by Report Type",
            ["PDF", "Excel", "JSON"],
            default=["PDF", "Excel", "JSON"]
        )
    with col2:
        date_range = st.date_input(
            "Date Range",
            [
                datetime.datetime.now() - datetime.timedelta(days=30),
                datetime.datetime.now()
            ]
        )
    
    # Convert report history to DataFrame for easier filtering
    if st.session_state.report_history:
        report_df = pd.DataFrame(st.session_state.report_history)
        
        # Apply filters
        if report_type_filter:
            report_df = report_df[report_df['type'].isin(report_type_filter)]
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            report_df = report_df[
                (report_df['date'].dt.date >= start_date) & 
                (report_df['date'].dt.date <= end_date)
            ]
        
        # Sort by date (newest first)
        report_df = report_df.sort_values('date', ascending=False)
        
        # Display reports
        if not report_df.empty:
            for i, row in report_df.iterrows():
                with st.expander(f"{row['title']} ({row['date'].strftime('%Y-%m-%d %H:%M')})"):
                    st.markdown(f"**Query:** {row['query']}")
                    st.markdown(f"**Report Type:** {row['type']}")
                    st.markdown(f"**Generated on:** {row['date'].strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Add mock download buttons
                    if row['type'] == 'PDF':
                        st.markdown('<a href="#" class="download-button">📄 Download PDF</a>', unsafe_allow_html=True)
                    elif row['type'] == 'Excel':
                        st.markdown('<a href="#" class="download-button">📊 Download Excel</a>', unsafe_allow_html=True)
                    elif row['type'] == 'JSON':
                        st.markdown('<a href="#" class="download-button">🔍 Download JSON</a>', unsafe_allow_html=True)
                    
                    # Add regenerate button
                    if st.button("Regenerate Report", key=f"regen_{i}"):
                        st.session_state.selected_query = row['query']
                        st.session_state.page = "💬 Chat with Data"
                        st.experimental_rerun()
        else:
            st.info("No reports match your filter criteria.")
    else:
        st.info("No reports have been generated yet. Use the Chat with Data page to create reports.")