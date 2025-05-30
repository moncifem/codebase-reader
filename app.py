"""
Streamlit application for the Codebase Reader & Analyzer.
Provides a user-friendly interface for analyzing codebases with AI.
"""

import streamlit as st
import os
import time
from pathlib import Path
from typing import Dict, Any, List
import traceback
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
    if 'current_codebase' not in st.session_state:
        st.session_state.current_codebase = None
    if 'indexing_in_progress' not in st.session_state:
        st.session_state.indexing_in_progress = False
    if 'indexed_files' not in st.session_state:
        st.session_state.indexed_files = []
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""


def get_analyzer() -> CodebaseAnalyzer:
    """Get or create the codebase analyzer."""
    if st.session_state.analyzer is None:
        try:
            st.session_state.analyzer = CodebaseAnalyzer()
        except Exception as e:
            st.error(f"Failed to initialize analyzer: {e}")
            st.stop()
    return st.session_state.analyzer


def setup_page():
    """Set up the Streamlit page configuration."""
    st.set_page_config(
        page_title=config.ui.page_title,
        page_icon=config.ui.page_icon,
        layout=config.ui.layout,
        initial_sidebar_state="expanded"
    )


def render_sidebar():
    """Render the sidebar with navigation and settings."""
    st.sidebar.title("🔍 Codebase Analyzer")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["📊 Dashboard", "📁 Index Codebase", "🔍 Search & Query", "⚙️ Settings"]
    )
    
    # Provider information
    analyzer = get_analyzer()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Current Providers")
    
    try:
        st.sidebar.info(f"🤖 Embeddings: {analyzer.embedding_manager.provider_type}")
        st.sidebar.info(f"🧠 LLM: {analyzer.llm_client.provider_type}")
    except Exception:
        st.sidebar.warning("Provider information unavailable")
    
    # Quick stats
    try:
        stats = analyzer.get_codebase_summary()
        if stats['total_chunks'] > 0:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### Quick Stats")
            st.sidebar.metric("Total Chunks", stats['total_chunks'])
            st.sidebar.metric("Files Indexed", stats['unique_files'])
            st.sidebar.metric("Languages", len(stats['languages']))
    except Exception:
        pass
    
    return page


def render_dashboard():
    """Render the main dashboard."""
    st.title("📊 Codebase Dashboard")
    st.markdown("Welcome to the Codebase Reader & Analyzer. Get insights into your code with AI.")
    
    analyzer = get_analyzer()
    
    try:
        summary = analyzer.get_codebase_summary()
        
        if summary['total_chunks'] == 0:
            st.info("🚀 No codebase indexed yet. Start by indexing a codebase using the sidebar.")
            st.markdown("### Getting Started")
            st.markdown("""
            1. **Index a Codebase**: Use the 'Index Codebase' page to scan and index your code
            2. **Search & Query**: Ask questions about your code or search for specific functionality
            3. **Settings**: Configure embedding and LLM providers
            """)
            return
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Chunks", summary['total_chunks'])
        
        with col2:
            st.metric("Files Indexed", summary['unique_files'])
        
        with col3:
            st.metric("Languages", len(summary['languages']))
        
        with col4:
            st.metric("Size (MB)", f"{summary['total_size_bytes'] / (1024*1024):.1f}")
        
        # Language breakdown
        if summary['languages']:
            st.markdown("### 📈 Language Distribution")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create a simple bar chart data
                lang_data = summary['languages']
                st.bar_chart(lang_data)
            
            with col2:
                st.markdown("**Languages:**")
                for lang, count in sorted(lang_data.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / summary['total_chunks']) * 100
                    st.markdown(f"- **{lang}**: {count} chunks ({percentage:.1f}%)")
        
        # Recent files
        st.markdown("### 📁 Indexed Files")
        files_info = summary.get('files', [])
        
        if files_info:
            # Sort by chunk count
            files_info.sort(key=lambda x: x['chunk_count'], reverse=True)
            
            # Display top files
            for file_info in files_info[:10]:
                file_path = file_info['file_path']
                file_name = os.path.basename(file_path)
                
                with st.expander(f"📄 {file_name} ({file_info['language']}) - {file_info['chunk_count']} chunks"):
                    st.code(file_path, language="text")
                    st.markdown(f"**Language**: {file_info['language']}")
                    st.markdown(f"**Chunks**: {file_info['chunk_count']}")
                    st.markdown(f"**Size**: {file_info.get('total_size', 0):,} bytes")
        
    except Exception as e:
        st.error(f"Error loading dashboard: {e}")
        st.exception(e)


def render_indexing_page():
    """Render the codebase indexing page."""
    st.title("📁 Index Codebase")
    st.markdown("Scan and index a codebase for AI-powered analysis.")
    
    analyzer = get_analyzer()
    
    # Directory selection
    st.markdown("### Select Codebase Directory")
    
    # Input methods
    input_method = st.radio(
        "How would you like to specify the directory?",
        ["📝 Type path", "📂 Browse current directory"]
    )
    
    codebase_path = None
    
    if input_method == "📝 Type path":
        codebase_path = st.text_input(
            "Enter the path to your codebase:",
            placeholder="/path/to/your/codebase",
            help="Enter the full path to the root directory of your codebase"
        )
    else:
        # Browse current directory
        current_dir = os.getcwd()
        st.info(f"Current directory: `{current_dir}`")
        
        try:
            subdirs = [d for d in os.listdir(current_dir) 
                      if os.path.isdir(os.path.join(current_dir, d)) and not d.startswith('.')]
            
            if subdirs:
                selected_dir = st.selectbox("Select a subdirectory:", [""] + subdirs)
                if selected_dir:
                    codebase_path = os.path.join(current_dir, selected_dir)
            else:
                st.warning("No subdirectories found in current directory.")
        except Exception as e:
            st.error(f"Error browsing directory: {e}")
    
    if codebase_path and os.path.exists(codebase_path):
        st.success(f"✅ Directory found: `{codebase_path}`")
        
        # Preview codebase
        try:
            preview_stats = analyzer.reader.get_codebase_stats(codebase_path)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files", preview_stats['total_files'])
            with col2:
                st.metric("Size (MB)", f"{preview_stats['total_size_mb']:.1f}")
            with col3:
                st.metric("Languages", len(preview_stats['languages']))
            
            if preview_stats['languages']:
                st.markdown("**Languages found:**")
                lang_text = ", ".join([f"{lang} ({count})" for lang, count in preview_stats['languages'].items()])
                st.markdown(lang_text)
        except Exception as e:
            st.warning(f"Could not preview codebase: {e}")
        
        # Indexing options
        st.markdown("### Indexing Options")
        
        col1, col2 = st.columns(2)
        with col1:
            update_existing = st.checkbox("Update existing files", value=True)
        with col2:
            clear_before_index = st.checkbox("Clear index before indexing", value=False)
        
        # Index button
        if st.button("🚀 Start Indexing", type="primary", disabled=st.session_state.indexing_in_progress):
            st.session_state.indexing_in_progress = True
            
            try:
                # Clear index if requested
                if clear_before_index:
                    with st.spinner("Clearing existing index..."):
                        analyzer.clear_index()
                        st.success("Index cleared!")
                
                # Start indexing
                with st.spinner("Indexing codebase... This may take a while."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Start indexing (this is a simplified progress indication)
                    status_text.text("Starting indexing process...")
                    progress_bar.progress(10)
                    
                    results = analyzer.index_codebase(codebase_path, update_existing=update_existing)
                    
                    progress_bar.progress(100)
                    status_text.text("Indexing completed!")
                
                # Show results
                st.success("🎉 Indexing completed successfully!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Files Processed", results['processed_files'])
                with col2:
                    st.metric("Chunks Added", results['added_chunks'])
                with col3:
                    st.metric("Files Updated", results['updated_files'])
                with col4:
                    st.metric("Files Skipped", results['skipped_files'])
                
                if results['errors']:
                    st.warning(f"⚠️ {len(results['errors'])} errors occurred during indexing:")
                    for error in results['errors']:
                        st.error(error)
                
                st.session_state.current_codebase = codebase_path
                st.session_state.indexed_files = results['indexed_files']
                
            except Exception as e:
                st.error(f"❌ Indexing failed: {e}")
                st.exception(e)
            finally:
                st.session_state.indexing_in_progress = False
    
    elif codebase_path:
        st.error(f"❌ Directory not found: `{codebase_path}`")


def render_search_page():
    """Render the search and query page."""
    st.title("🔍 Search & Query")
    st.markdown("Search your codebase and ask AI questions about your code.")
    
    analyzer = get_analyzer()
    
    # Check if codebase is indexed
    try:
        summary = analyzer.get_codebase_summary()
        if summary['total_chunks'] == 0:
            st.warning("⚠️ No codebase indexed yet. Please index a codebase first.")
            return
    except Exception:
        st.error("❌ Unable to check codebase status.")
        return
    
    # Query interface
    st.markdown("### 🤖 Ask AI About Your Code")
    
    query = st.text_area(
        "Enter your question:",
        placeholder="E.g., How does authentication work in this codebase?",
        height=100
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        context_chunks = st.slider("Number of code chunks for context", 1, 10, 5)
    
    if st.button("🧠 Ask AI", type="primary") and query:
        with st.spinner("Analyzing codebase and generating response..."):
            try:
                response = analyzer.ask_question(query, n_context_chunks=context_chunks)
                st.markdown("### 💬 AI Response")
                st.markdown(response)
            except Exception as e:
                st.error(f"Error generating response: {e}")
    
    st.markdown("---")
    
    # Code search
    st.markdown("### 🔍 Search Code")
    
    search_query = st.text_input(
        "Search for code:",
        placeholder="E.g., user authentication, database connection, error handling"
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        num_results = st.slider("Number of results", 1, 20, 10)
    with col2:
        language_filter = st.selectbox("Language filter", ["All"] + list(summary['languages'].keys()))
    with col3:
        file_filter = st.text_input("File path contains:", placeholder="e.g., models/")
    
    if st.button("🔍 Search") and search_query:
        with st.spinner("Searching codebase..."):
            try:
                # Prepare filters
                lang_filter = None if language_filter == "All" else language_filter
                file_filter = None if not file_filter.strip() else file_filter.strip()
                
                results = analyzer.search_code(
                    query=search_query,
                    n_results=num_results,
                    language_filter=lang_filter,
                    file_path_filter=file_filter
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
                        
                        with st.expander(f"#{i+1} {file_name} ({language}) - Similarity: {similarity:.2f}"):
                            st.markdown(f"**File**: `{file_path}`")
                            st.markdown(f"**Lines**: {start_line}-{end_line}")
                            st.markdown(f"**Language**: {language}")
                            st.markdown(f"**Similarity**: {similarity:.3f}")
                            
                            st.markdown("**Code:**")
                            st.code(result['content'], language=language)
                            
                            # Action buttons
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
                st.exception(e)


def render_settings_page():
    """Render the settings page."""
    st.title("⚙️ Settings")
    st.markdown("Configure embedding and LLM providers.")
    
    analyzer = get_analyzer()
    
    # Embedding provider settings
    st.markdown("### 🤖 Embedding Provider")
    
    available_embedding_providers = analyzer.embedding_manager.get_available_providers()
    current_embedding = analyzer.embedding_manager.provider_type
    
    if available_embedding_providers:
        new_embedding_provider = st.selectbox(
            "Select embedding provider:",
            available_embedding_providers,
            index=available_embedding_providers.index(current_embedding) if current_embedding in available_embedding_providers else 0
        )
        
        if new_embedding_provider != current_embedding:
            if st.button("Switch Embedding Provider"):
                if analyzer.switch_embedding_provider(new_embedding_provider):
                    st.success(f"Switched to {new_embedding_provider}")
                    st.rerun()
                else:
                    st.error("Failed to switch embedding provider")
    else:
        st.warning("No embedding providers available. Check your configuration and API keys.")
    
    # LLM provider settings
    st.markdown("### 🧠 LLM Provider")
    
    available_llm_providers = analyzer.llm_client.get_available_providers()
    current_llm = analyzer.llm_client.provider_type
    
    if available_llm_providers:
        new_llm_provider = st.selectbox(
            "Select LLM provider:",
            available_llm_providers,
            index=available_llm_providers.index(current_llm) if current_llm in available_llm_providers else 0
        )
        
        if new_llm_provider != current_llm:
            if st.button("Switch LLM Provider"):
                if analyzer.switch_llm_provider(new_llm_provider):
                    st.success(f"Switched to {new_llm_provider}")
                    st.rerun()
                else:
                    st.error("Failed to switch LLM provider")
    else:
        st.warning("No LLM providers available. Check your configuration and API keys.")
    
    # Configuration info
    st.markdown("### 📋 Configuration")
    
    with st.expander("View Current Configuration"):
        st.json({
            "embedding_provider": current_embedding,
            "llm_provider": current_llm,
            "vector_db": {
                "type": config.vector_db.type,
                "persist_directory": config.vector_db.persist_directory,
                "collection_name": config.vector_db.collection_name
            },
            "chunking": {
                "chunk_size": config.chunking.chunk_size,
                "chunk_overlap": config.chunking.chunk_overlap,
                "max_file_size_mb": config.chunking.max_file_size_mb
            }
        })
    
    # Clear index option
    st.markdown("### 🗑️ Manage Index")
    st.warning("⚠️ Clearing the index will remove all indexed code chunks.")
    
    if st.button("Clear Index", type="secondary"):
        if analyzer.clear_index():
            st.success("Index cleared successfully!")
        else:
            st.error("Failed to clear index")


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
            render_settings_page()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)


if __name__ == "__main__":
    main() 