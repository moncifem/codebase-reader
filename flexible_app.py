"""
Flexible Streamlit application for the Codebase Reader & Analyzer.
Gracefully handles missing providers and provides clear status information.
"""

import streamlit as st
import os
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Set up the environment
from dotenv import load_dotenv
load_dotenv()

# Import our application components
from src.config_manager import config
from src.codebase_analyzer import CodebaseAnalyzer


def init_session_state():
    """Initialize Streamlit session state."""
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'indexed_files' not in st.session_state:
        st.session_state.indexed_files = []
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""


def setup_page():
    """Set up the Streamlit page configuration."""
    st.set_page_config(
        page_title=config.ui.page_title,
        page_icon=config.ui.page_icon,
        layout=config.ui.layout,
        initial_sidebar_state="expanded"
    )


def get_or_create_analyzer():
    """Get or create the analyzer instance."""
    if st.session_state.analyzer is None:
        with st.spinner("Initializing Codebase Analyzer..."):
            st.session_state.analyzer = CodebaseAnalyzer()
    return st.session_state.analyzer


def render_sidebar():
    """Render the sidebar with navigation and status."""
    st.sidebar.title("🔍 Codebase Analyzer")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["📊 Dashboard", "📁 Index Codebase", "🔍 Search & Query", "⚙️ Settings", "🔧 Provider Status"]
    )
    
    # Try to get analyzer status
    try:
        analyzer = get_or_create_analyzer()
        status = analyzer.get_provider_status()
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Provider Status")
        
        # Embedding status
        if status['current_embedding']:
            st.sidebar.success(f"🔢 Embeddings: {status['current_embedding']['display_name']}")
        else:
            st.sidebar.error("🔢 Embeddings: Not available")
        
        # LLM status
        if status['current_llm']:
            st.sidebar.success(f"🤖 LLM: {status['current_llm']['display_name']}")
        else:
            st.sidebar.error("🤖 LLM: Not available")
        
        # Vector store status
        if status['vector_store_available']:
            st.sidebar.success("💾 Vector Store: Ready")
        else:
            st.sidebar.error("💾 Vector Store: Not available")
        
    except Exception as e:
        st.sidebar.error(f"❌ Analyzer Error: {str(e)}")
    
    return page


def render_dashboard():
    """Render the main dashboard with provider status."""
    st.title("📊 Codebase Dashboard")
    st.markdown("Welcome to the Flexible Codebase Reader & Analyzer")
    
    analyzer = get_or_create_analyzer()
    
    # Provider status overview
    status = analyzer.get_provider_status()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if status['current_embedding']:
            st.metric("Embedding Provider", status['current_embedding']['display_name'], "✅ Active")
        else:
            st.metric("Embedding Provider", "None", "❌ Missing")
    
    with col2:
        if status['current_llm']:
            st.metric("LLM Provider", status['current_llm']['display_name'], "✅ Active")
        else:
            st.metric("LLM Provider", "None", "❌ Missing")
    
    with col3:
        if status['vector_store_available']:
            st.metric("Vector Store", "ChromaDB", "✅ Ready")
        else:
            st.metric("Vector Store", "Unavailable", "❌ Error")
    
    # Codebase summary if available
    if status['vector_store_available']:
        try:
            summary = analyzer.get_codebase_summary()
            
            st.markdown("### 📈 Codebase Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Chunks", summary['total_chunks'])
            
            with col2:
                st.metric("Files Indexed", summary['unique_files'])
            
            with col3:
                st.metric("Languages", len(summary['languages']))
            
            with col4:
                st.metric("Size (MB)", f"{summary['total_size_bytes'] / (1024*1024):.1f}")
            
            # Language distribution
            if summary['languages']:
                st.markdown("### 📊 Language Distribution")
                lang_df = pd.DataFrame(
                    list(summary['languages'].items()),
                    columns=['Language', 'Chunks']
                )
                st.bar_chart(lang_df.set_index('Language'))
            
        except Exception as e:
            st.warning(f"Could not load codebase summary: {e}")
    
    else:
        st.info("👋 Vector store not available. Please check your provider configuration in Settings.")


def render_provider_status():
    """Render detailed provider status page."""
    st.title("🔧 Provider Status")
    st.markdown("Detailed information about available and configured providers")
    
    analyzer = get_or_create_analyzer()
    status = analyzer.get_provider_status()
    
    # Embedding providers
    st.markdown("### 🔢 Embedding Providers")
    
    for provider in status['embedding_providers']:
        with st.expander(f"{provider['display_name']} - {'✅ Available' if provider['available'] else '❌ Unavailable'}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Name**: {provider['name']}")
                st.write(f"**Status**: {'Available' if provider['available'] else 'Unavailable'}")
                st.write(f"**Requires API Key**: {'Yes' if provider['requires_api_key'] else 'No'}")
            
            with col2:
                if provider['error']:
                    st.error(f"**Error**: {provider['error']}")
                else:
                    st.success("**Status**: Ready to use")
    
    # LLM providers
    st.markdown("### 🤖 LLM Providers")
    
    for provider in status['llm_providers']:
        with st.expander(f"{provider['display_name']} - {'✅ Available' if provider['available'] else '❌ Unavailable'}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Name**: {provider['name']}")
                st.write(f"**Status**: {'Available' if provider['available'] else 'Unavailable'}")
                st.write(f"**Requires API Key**: {'Yes' if provider['requires_api_key'] else 'No'}")
            
            with col2:
                if provider['error']:
                    st.error(f"**Error**: {provider['error']}")
                else:
                    st.success("**Status**: Ready to use")
    
    # Environment variables
    st.markdown("### 🔑 Environment Variables")
    
    env_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            st.success(f"✅ {var}: Configured")
        else:
            st.error(f"❌ {var}: Not set")


def render_settings():
    """Render the settings page with provider switching."""
    st.title("⚙️ Settings")
    st.markdown("Configure providers and system settings")
    
    analyzer = get_or_create_analyzer()
    status = analyzer.get_provider_status()
    
    # Provider switching
    st.markdown("### 🔄 Switch Providers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Embedding Provider")
        
        available_embedding = [p for p in status['embedding_providers'] if p['available']]
        if available_embedding:
            current_embedding = status['current_embedding']['name'] if status['current_embedding'] else None
            
            provider_names = [p['name'] for p in available_embedding]
            current_index = provider_names.index(current_embedding) if current_embedding in provider_names else 0
            
            selected_embedding = st.selectbox(
                "Select embedding provider:",
                provider_names,
                index=current_index,
                format_func=lambda x: next(p['display_name'] for p in available_embedding if p['name'] == x)
            )
            
            if st.button("Switch Embedding Provider"):
                if analyzer.switch_embedding_provider(selected_embedding):
                    st.success("✅ Embedding provider switched successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to switch embedding provider")
        else:
            st.warning("No embedding providers available")
    
    with col2:
        st.markdown("#### LLM Provider")
        
        available_llm = [p for p in status['llm_providers'] if p['available']]
        if available_llm:
            current_llm = status['current_llm']['name'] if status['current_llm'] else None
            
            provider_names = [p['name'] for p in available_llm]
            current_index = provider_names.index(current_llm) if current_llm in provider_names else 0
            
            selected_llm = st.selectbox(
                "Select LLM provider:",
                provider_names,
                index=current_index,
                format_func=lambda x: next(p['display_name'] for p in available_llm if p['name'] == x)
            )
            
            if st.button("Switch LLM Provider"):
                if analyzer.switch_llm_provider(selected_llm):
                    st.success("✅ LLM provider switched successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to switch LLM provider")
        else:
            st.warning("No LLM providers available")
    
    # Configuration display
    st.markdown("### 📋 Current Configuration")
    
    with st.expander("View Configuration"):
        st.json({
            "embedding_provider": status['current_embedding']['name'] if status['current_embedding'] else None,
            "llm_provider": status['current_llm']['name'] if status['current_llm'] else None,
            "vector_store_available": status['vector_store_available'],
            "vector_db_config": {
                "type": config.vector_db.type,
                "persist_directory": config.vector_db.persist_directory,
                "collection_name": config.vector_db.collection_name,
                "distance_metric": config.vector_db.distance_metric
            }
        })


def render_indexing_page():
    """Render the codebase indexing page."""
    st.title("📁 Index Codebase")
    st.markdown("Scan and index a codebase for AI-powered analysis")
    
    analyzer = get_or_create_analyzer()
    status = analyzer.get_provider_status()
    
    # Check if we can index
    if not status['current_embedding'] or not status['vector_store_available']:
        st.error("❌ Cannot index codebase: Missing embedding provider or vector store")
        st.info("Please configure an embedding provider in the Settings page")
        return
    
    # Initialize session state for path input and verification status
    if 'codebase_path_input' not in st.session_state:
        st.session_state.codebase_path_input = st.session_state.get('last_verified_path', "") # Pre-fill with last verified
    if 'is_path_verified' not in st.session_state:
        st.session_state.is_path_verified = False # Tracks if the CURRENT input is verified
    if 'last_verified_path' not in st.session_state:
        st.session_state.last_verified_path = ""

    # Directory selection
    st.markdown("### 📂 Select Directory")
    
    # Display ignore patterns for transparency
    with st.expander("📋 Files & Directories Being Ignored", expanded=False):
        st.markdown("The following patterns will be **excluded** from indexing:")
        
        try:
            ignore_patterns = config.ignore_patterns
            if ignore_patterns:
                col_patterns1, col_patterns2 = st.columns(2)
                mid_point = len(ignore_patterns) // 2
                
                with col_patterns1:
                    for pattern in ignore_patterns[:mid_point]:
                        st.markdown(f"❌ `{pattern}`")
                
                with col_patterns2:
                    for pattern in ignore_patterns[mid_point:]:
                        st.markdown(f"❌ `{pattern}`")
            else:
                st.info("No ignore patterns configured.")
        except Exception as e:
            st.warning(f"Could not load ignore patterns: {e}")
        
        st.markdown("*These patterns are configured in `config.yaml` under `ignore_patterns`.*")
    
    col_path, col_verify = st.columns([3,1])

    with col_path:
        # Use a temporary variable for the input to detect changes
        current_input_path = st.text_input(
            "Enter the path to your codebase:",
            value=st.session_state.codebase_path_input, # Controlled component
            placeholder="/path/to/your/codebase",
            help="Enter the full path to the root directory of your codebase. Click 'Verify Path' to confirm.",
            key="path_input_widget"
        )
        # If path in widget changes, update session state and mark as unverified
        if current_input_path != st.session_state.codebase_path_input:
            st.session_state.codebase_path_input = current_input_path
            st.session_state.is_path_verified = False # Path changed, requires re-verification
            st.rerun() # Rerun to update UI based on new unverified state

    with col_verify:
        st.write("&nbsp;") # Add some space for alignment
        verify_button = st.button("👁️ Verify Path", key="verify_path_button")

    if verify_button:
        path_to_verify = st.session_state.codebase_path_input
        if path_to_verify:
            if os.path.isdir(path_to_verify):
                st.session_state.is_path_verified = True
                st.session_state.last_verified_path = path_to_verify # Remember for pre-filling
                st.success(f"✅ Directory verified: `{path_to_verify}`")
                try:
                    st.markdown("**Sample contents (first 10 items):**")
                    contents = os.listdir(path_to_verify)[:10]
                    if not contents:
                        st.info("Directory is empty.")
                    else:
                        for item in contents:
                            item_path = os.path.join(path_to_verify, item)
                            if os.path.isdir(item_path):
                                st.markdown(f"📁 `{item}/`")
                            else:
                                st.markdown(f"📄 `{item}`")
                except Exception as e:
                    st.warning(f"Could not list directory contents: {e}")
            else:
                st.session_state.is_path_verified = False
                st.error(f"❌ Path is not a valid directory: `{path_to_verify}`")
        else:
            st.session_state.is_path_verified = False
            st.warning("Please enter a path to verify.")

    # Indexing section - only show if the current path_input_widget value is verified
    if st.session_state.is_path_verified and st.session_state.codebase_path_input == st.session_state.last_verified_path:
        actual_codebase_path = st.session_state.last_verified_path
        st.markdown("---")
        st.markdown(f"**Selected for indexing:** `{actual_codebase_path}`")
        
        col_opts, col_button = st.columns([3,1])
        with col_opts:
            update_existing = st.checkbox("Update existing files if already indexed", value=True, key="update_existing_cb")
        
        with col_button:
            st.write("&nbsp;") 
            if st.button("🚀 Start Indexing", type="primary", key="start_indexing_button"):
                with st.spinner(f"Indexing `{actual_codebase_path}`... This may take a while."):
                    results = analyzer.index_codebase(actual_codebase_path, update_existing=update_existing)
                
                if results.get('success', False):
                    st.success("🎉 Indexing completed successfully!")
                    metrics_cols = st.columns(4)
                    metrics_cols[0].metric("Files Processed", results.get('processed_files', 0))
                    metrics_cols[1].metric("Chunks Added", results.get('added_chunks', 0))
                    metrics_cols[2].metric("Updated Files", results.get('updated_files', 0))
                    metrics_cols[3].metric("Skipped Files", results.get('skipped_files', 0))
                    
                    if results.get('errors'):
                        st.warning(f"⚠️ {len(results['errors'])} errors occurred during indexing:")
                        for error_msg in results['errors']:
                            st.error(error_msg)
                    st.session_state.indexed_files = results.get('indexed_files', [])
                else:
                    st.error(f"❌ Indexing failed: {results.get('error', 'Unknown error')}")
    
    elif st.session_state.codebase_path_input and not st.session_state.is_path_verified:
        st.info("Path has changed or is not yet verified. Please click 'Verify Path' to confirm the directory before indexing.")


def render_search_page():
    """Render the search and query page."""
    st.title("🔍 Search & Query")
    st.markdown("Search your codebase and ask AI questions about your code")
    
    analyzer = get_or_create_analyzer()
    status = analyzer.get_provider_status()
    
    # Check if we have a codebase indexed
    if not status['vector_store_available']:
        st.warning("⚠️ Vector store not available. Please check your configuration.")
        return
    
    try:
        summary = analyzer.get_codebase_summary()
        if summary['total_chunks'] == 0:
            st.warning("⚠️ No codebase indexed yet. Please index a codebase first.")
            return
    except Exception:
        st.error("❌ Unable to check codebase status.")
        return
    
    # AI Questions section
    if status['current_llm']:
        st.markdown("### 🤖 Ask AI About Your Code")
        
        query = st.text_area(
            "Enter your question:",
            value=st.session_state.last_query,
            placeholder="E.g., How does authentication work in this codebase?",
            height=100
        )
        
        context_chunks = st.slider("Number of code chunks for context", 1, 10, 5)
        
        if st.button("🧠 Ask AI", type="primary") and query:
            st.session_state.last_query = query
            with st.spinner("Analyzing codebase and generating response..."):
                try:
                    response = analyzer.ask_question(query, n_context_chunks=context_chunks)
                    st.markdown("### 💬 AI Response")
                    st.markdown(response)
                except Exception as e:
                    st.error(f"Error generating response: {e}")
    else:
        st.info("🤖 LLM provider not available. You can still search for code below.")
    
    st.markdown("---")
    
    # Code search
    st.markdown("### 🔍 Search Code")
    
    search_query = st.text_input(
        "Search for code:",
        placeholder="E.g., user authentication, database connection, error handling"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        num_results = st.slider("Number of results", 1, 20, 10)
    with col2:
        language_filter = st.selectbox("Language filter", ["All"] + list(summary['languages'].keys()))
    
    if st.button("🔍 Search") and search_query:
        with st.spinner("Searching codebase..."):
            try:
                lang_filter = None if language_filter == "All" else language_filter
                
                results = analyzer.search_code(
                    query=search_query,
                    n_results=num_results,
                    language_filter=lang_filter
                )
                
                if results:
                    st.markdown(f"### 📋 Search Results ({len(results)} found)")
                    
                    for i, result in enumerate(results):
                        metadata = result['metadata']
                        similarity = result['similarity']
                        
                        file_path = metadata['file_path']
                        file_name = os.path.basename(file_path)
                        language = metadata['language']
                        start_line = metadata['start_line']
                        end_line = metadata['end_line']
                        
                        with st.expander(f"#{i+1} {file_name} ({language}) - Similarity: {similarity:.3f}"):
                            st.markdown(f"**File**: `{file_path}`")
                            st.markdown(f"**Lines**: {start_line}-{end_line}")
                            st.markdown(f"**Language**: {language}")
                            
                            st.markdown("**Code:**")
                            st.code(result['content'], language=language)
                            
                            # AI analysis buttons (only if LLM is available)
                            if status['current_llm']:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    if st.button(f"📖 Explain", key=f"explain_{result['id']}"):
                                        with st.spinner("Generating explanation..."):
                                            explanation = analyzer.explain_code_chunk(result['id'])
                                            if explanation:
                                                st.markdown("**AI Explanation:**")
                                                st.markdown(explanation)
                                
                                with col2:
                                    if st.button(f"💡 Suggest Improvements", key=f"improve_{result['id']}"):
                                        with st.spinner("Generating suggestions..."):
                                            suggestions = analyzer.suggest_improvements(result['id'])
                                            if suggestions:
                                                st.markdown("**AI Suggestions:**")
                                                st.markdown(suggestions)
                                
                                with col3:
                                    if st.button(f"🔒 Security Check", key=f"security_{result['id']}"):
                                        with st.spinner("Analyzing security..."):
                                            security_analysis = analyzer.find_security_issues(result['id'])
                                            if security_analysis:
                                                st.markdown("**Security Analysis:**")
                                                st.markdown(security_analysis)
                else:
                    st.info("No results found for your search query.")
                    
            except Exception as e:
                st.error(f"Search failed: {e}")


def main():
    """Main application entry point."""
    setup_page()
    init_session_state()
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Render the selected page
    try:
        if page == "📊 Dashboard":
            render_dashboard()
        elif page == "📁 Index Codebase":
            render_indexing_page()
        elif page == "🔍 Search & Query":
            render_search_page()
        elif page == "⚙️ Settings":
            render_settings()
        elif page == "🔧 Provider Status":
            render_provider_status()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)


if __name__ == "__main__":
    main() 