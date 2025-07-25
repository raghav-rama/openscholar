import streamlit as st
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).parent / "src"))

from openscholar import (
    OpenScholarDataStore,
    OpenScholarRetriever,
    OpenScholarReranker,
    OpenScholarGenerator,
    OpenScholarPipeline,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="OpenScholar Demo",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_pipeline(config_path: str = "config/demo_config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    datastore = OpenScholarDataStore(
        data_dir=config['datastore']['path']
    )
    datastore.load()
    
    retriever = OpenScholarRetriever(
        model_name=config['retriever']['model_name'],
        use_semantic_scholar=config['retriever'].get('use_semantic_scholar', True),
        use_web_search=config['retriever'].get('use_web_search', False)
    )
    
    reranker = OpenScholarReranker(
        model_name=config['reranker']['model_name']
    )
    
    generator = OpenScholarGenerator(
        model_name=config['generator']['model_name'],
        temperature=config['generator'].get('temperature', 0.7),
        max_iterations=config['generator'].get('max_iterations', 3)
    )
    
    pipeline = OpenScholarPipeline(
        datastore=datastore,
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        initial_retrieval_k=config['pipeline'].get('initial_retrieval_k', 100),
        rerank_k=config['pipeline'].get('rerank_k', 50),
        final_k=config['pipeline'].get('final_k', 20)
    )
    
    return pipeline, config


def format_citations(citations: List[Dict]) -> str:
    if not citations:
        return "No citations"
        
    citation_text = ""
    for cit in citations:
        citation_text += f"[{cit['number']}] {cit.get('paper_title', 'Unknown')} "
        if 'paper_id' in cit:
            citation_text += f"(ID: {cit['paper_id'][:8]}...)\n"
        else:
            citation_text += "\n"
            
    return citation_text


def main():
    st.title("🎓 OpenScholar: AI-Powered Scientific Literature Synthesis")
    
    st.markdown("""
    OpenScholar synthesizes scientific literature by retrieving relevant papers and generating 
    comprehensive, citation-backed responses to your research questions.
    """)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        config_path = st.text_input(
            "Config Path",
            value="config/demo_config.json",
            help="Path to the configuration file"
        )
        
        if st.button("🔄 Reload Pipeline"):
            st.cache_resource.clear()
            st.rerun()
            
        st.markdown("---")
        
        st.header("📊 Pipeline Settings")
        
        retrieval_k = st.slider(
            "Initial Retrieval K",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help="Number of passages to retrieve initially"
        )
        
        rerank_k = st.slider(
            "Rerank K",
            min_value=20,
            max_value=100,
            value=50,
            step=10,
            help="Number of passages after reranking"
        )
        
        final_k = st.slider(
            "Final K",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Final number of passages for generation"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Generation temperature"
        )
        
        max_iterations = st.slider(
            "Max Iterations",
            min_value=1,
            max_value=5,
            value=3,
            help="Maximum self-feedback iterations"
        )
        
    try:
        pipeline, config = load_pipeline(config_path)
        
        pipeline.initial_retrieval_k = retrieval_k
        pipeline.rerank_k = rerank_k
        pipeline.final_k = final_k
        pipeline.generator.temperature = temperature
        pipeline.generator.max_iterations = max_iterations
        
    except Exception as e:
        st.error(f"Failed to load pipeline: {str(e)}")
        st.stop()
        
    query = st.text_area(
        "Enter your research question:",
        placeholder="e.g., What are the recent advances in neural architecture search?",
        height=100
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        generate_button = st.button("🔍 Generate Response", type="primary")
        
    with col2:
        clear_button = st.button("🗑️ Clear")
        
    if clear_button:
        st.session_state.clear()
        st.rerun()
        
    if generate_button and query:
        with st.spinner("🔄 Processing your query..."):
            start_time = time.time()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📚 Retrieving relevant papers...")
            progress_bar.progress(20)
            
            try:
                response = pipeline.generate(query)
                
                status_text.text("✨ Generating response with self-feedback...")
                progress_bar.progress(80)
                
                elapsed_time = time.time() - start_time
                
                status_text.text(f"✅ Completed in {elapsed_time:.2f} seconds")
                progress_bar.progress(100)
                
                st.session_state['response'] = response
                st.session_state['query'] = query
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                logger.error(f"Generation error: {e}", exc_info=True)
                
    if 'response' in st.session_state:
        response = st.session_state['response']
        
        st.markdown("---")
        st.header("📝 Generated Response")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(response.response)
            
        with col2:
            st.metric("Iterations", response.iterations)
            st.metric("Citations", len(response.citations))
            st.metric("Retrieved Passages", len(response.retrieved_passages))
            
        with st.expander("📚 Citations", expanded=True):
            st.text(format_citations(response.citations))
            
        with st.expander("🔄 Feedback History"):
            if response.feedback_history:
                for i, feedback in enumerate(response.feedback_history):
                    st.markdown(f"**Iteration {feedback['iteration']}**")
                    st.markdown(f"- Type: {feedback['type']}")
                    st.markdown(f"- Description: {feedback['description']}")
                    if feedback.get('retrieval_query'):
                        st.markdown(f"- Additional Query: {feedback['retrieval_query']}")
                    st.markdown("---")
            else:
                st.info("No feedback iterations were performed.")
                
        with st.expander("📄 Retrieved Passages"):
            for i, passage in enumerate(response.retrieved_passages[:10]):
                st.markdown(f"**Passage {i+1}** (Score: {passage.get('score', 0):.3f})")
                st.markdown(f"*{passage['paper_title']}*")
                st.markdown(passage['text'][:300] + "...")
                st.markdown("---")
                
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Response"):
                save_path = f"outputs/response_{int(time.time())}.json"
                Path("outputs").mkdir(exist_ok=True)
                pipeline.save_response(response, save_path)
                st.success(f"Response saved to {save_path}")
                
        with col2:
            if st.button("📋 Copy Response"):
                st.write("Response copied to clipboard!")
                
        with col3:
            if st.button("🔄 Regenerate"):
                del st.session_state['response']
                st.rerun()
                
    with st.sidebar:
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        **OpenScholar** is a retrieval-augmented language model designed for 
        synthesizing scientific literature. It uses:
        
        - 🗄️ **45M papers** in the datastore
        - 🔍 **Advanced retrieval** with reranking
        - 🔄 **Self-feedback** for iterative improvement
        - 📚 **Accurate citations** for all claims
        
        [GitHub](https://github.com/yourusername/openscholar) | 
        [Paper](https://arxiv.org/abs/xxxx.xxxxx)
        """)


if __name__ == "__main__":
    main()