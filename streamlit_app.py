"""
Streamlit Frontend for RAG System
A beautiful web interface for interacting with the RAG system without modifying the core project.
"""

import streamlit as st
import subprocess
import json
import time
import os
import sys
from pathlib import Path
import pandas as pd

# Add the src directory to the path to import RAG components
sys.path.append(str(Path(__file__).parent / "src"))

# Page configuration
st.set_page_config(
    page_title="RAG System - AI-Powered Document Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .response-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }
    .query-response {
        font-size: 14px !important;
        line-height: 1.4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .compact-stats {
        font-size: 0.85rem;
        padding: 0.5rem;
    }
    .compact-queries {
        font-size: 0.8rem;
        margin-bottom: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

def run_rag_query(query, k=5, max_tokens=200, temperature=0.7):
    """Run a query through the RAG system using the CLI"""
    try:
        cmd = [
            "python", "rag_cli.py",
            "--query", query,
            "--k", str(k),
            "--max-tokens", str(max_tokens),
            "--temperature", str(temperature)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        if result.returncode == 0:
            return result.stdout, None
        else:
            return None, result.stderr
    except Exception as e:
        return None, str(e)

def get_system_info():
    """Get system information and status"""
    info = {}
    
    # Check if models exist
    model_path = Path("models/Llama-3.1-8B-Instruct-int4-genai")
    info["model_available"] = model_path.exists()
    info["model_path"] = str(model_path)
    
    # Check vector store
    vector_store_path = Path("data/processed_data/vector_store")
    info["vector_store_available"] = vector_store_path.exists()
    info["vector_store_path"] = str(vector_store_path)
    
    # Check GPU availability
    try:
        import torch
        info["gpu_available"] = torch.cuda.is_available()
        if info["gpu_available"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    except:
        info["gpu_available"] = False
    
    return info

def process_query(query, k_chunks, max_tokens, temperature):
    """Process a query and return the response with timing"""
    import time
    
    start_time = time.time()
    with st.spinner("Processing your query..."):
        response, error = run_rag_query(query, k_chunks, max_tokens, temperature)
        
        if response:
            # Parse the response to extract the answer
            lines = response.split('\n')
            answer_started = False
            answer_lines = []
            
            for line in lines:
                if "Response:" in line:
                    answer_started = True
                    continue
                elif "================================================================================" in line and answer_started:
                    break
                elif answer_started and line.strip() and not line.startswith("  Generating embeddings") and not line.startswith("Using FAISS") and not line.startswith("2025-") and not line.startswith("INFO -"):
                    answer_lines.append(line.strip())
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if answer_lines:
                answer = '\n'.join(answer_lines)
                return answer, None, processing_time
            else:
                # Debug: Show the raw response for troubleshooting
                with st.expander("🔍 Debug: Raw Response", expanded=False):
                    st.text(response)
                return None, "Could not parse response from RAG system", processing_time
        else:
            end_time = time.time()
            processing_time = end_time - start_time
            return None, f"Error running query: {error}", processing_time

def format_metric_name(metric):
    """Format metric names for better display"""
    name_mapping = {
        'Cpu Usage': '🖥️ CPU Usage',
        'Memory Usage': '💾 Memory Usage',
        'Gpu Usage': '🚀 GPU Usage',
        'Gpu Memory': '📊 GPU Memory',
        'Gpu Memory Gb': '💿 GPU Memory',
        'Inference Latency': '⏱️ Inference Latency',
        'Throughput': '🔄 Throughput',
        'Total Inference Count': '📈 Total Inferences',
        'Device': '🔧 Device'
    }
    return name_mapping.get(metric, metric)

def format_metric_value(raw_value, format_rule):
    """Format metric values with appropriate units and precision"""
    try:
        # Handle non-numeric values (like 'CPU' for device)
        if not isinstance(raw_value, (int, float)):
            try:
                numeric_value = float(raw_value)
            except (ValueError, TypeError):
                return str(raw_value)  # Return as-is for non-numeric values
        else:
            numeric_value = raw_value
        
        # Apply precision formatting
        if format_rule['precision'] == 0:
            formatted = f"{int(numeric_value)}"
        else:
            formatted = f"{numeric_value:.{format_rule['precision']}f}"
        
        # Add unit if specified
        if format_rule['unit']:
            return f"{formatted} {format_rule['unit']}"
        else:
            return formatted
            
    except Exception:
        return str(raw_value)

def get_metric_color(metric, value, default_color):
    """Determine appropriate color based on metric type and value"""
    try:
        numeric_value = float(value) if isinstance(value, str) else value
        
        # Color coding based on performance implications
        if metric in ['Cpu Usage', 'Memory Usage', 'Gpu Usage', 'Gpu Memory']:
            if numeric_value > 80:
                return '#dc3545'  # Red for high usage
            elif numeric_value > 60:
                return '#ffc107'  # Yellow for medium usage
            else:
                return '#28a745'  # Green for low usage
        elif metric == 'Inference Latency':
            if numeric_value > 1000:  # More than 1 second
                return '#dc3545'  # Red for high latency
            elif numeric_value > 500:  # More than 500ms
                return '#ffc107'  # Yellow for medium latency
            else:
                return '#28a745'  # Green for low latency
        elif metric == 'Throughput':
            if numeric_value > 10:
                return '#28a745'  # Green for high throughput
            elif numeric_value > 5:
                return '#ffc107'  # Yellow for medium throughput
            else:
                return '#dc3545'  # Red for low throughput
        else:
            return default_color
            
    except (ValueError, TypeError):
        return default_color

def parse_performance_output(output):
    """Parse performance output and extract metrics with improved error handling"""
    lines = output.split('\n')
    metrics = {}
    
    # Look for the final statistics table
    in_final_stats = False
    for line in lines:
        if "Final statistics:" in line:
            in_final_stats = True
            continue
        elif in_final_stats and "| Metric | Value |" in line:
            continue
        elif in_final_stats and line.strip().startswith("+") and "+" in line and "-" in line:
            # Skip separator lines
            continue
        elif in_final_stats and "|" in line and not line.strip().startswith("+"):
            # Parse metric line
            parts = line.split("|")
            if len(parts) >= 3:
                metric = parts[1].strip()
                value = parts[2].strip()
                if metric and value and metric != "Metric" and value != "Value":
                    # Clean up the value (remove any extra formatting)
                    cleaned_value = value.replace('%', '').replace('GB', '').replace('ms', '').replace('tokens/s', '').strip()
                    try:
                        # Try to convert to float for numeric values
                        if cleaned_value and cleaned_value != '-' and cleaned_value != 'N/A':
                            metrics[metric] = float(cleaned_value) if '.' in cleaned_value else int(cleaned_value)
                        else:
                            metrics[metric] = value  # Keep original if conversion fails
                    except (ValueError, TypeError):
                        metrics[metric] = value  # Keep original value if not numeric
        elif in_final_stats and line.strip() == "":
            # Empty line might indicate end of table
            break
    
    return metrics

def display_performance_metrics(metrics):
    """Display performance metrics using Streamlit's native components for better presentation"""
    if not metrics:
        st.warning("No performance data available")
        return
    
    # Create a nice header
    st.markdown("""
    <div style='background: linear-gradient(90deg, #e8f5e8 0%, #f0f9ff 100%); 
                padding: 1rem; border-radius: 0.5rem; border: 1px solid #28a745; 
                margin: 0.5rem 0; text-align: center;'>
        <h4 style='color: #155724; margin: 0; font-size: 1.1rem;'>
            📊 Performance Test Results
        </h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Define metric formatting and grouping
    metric_groups = {
        'System Resources': {
            'Cpu Usage': {'icon': '🖥️', 'label': 'CPU Usage', 'unit': '%', 'precision': 1},
            'Memory Usage': {'icon': '💾', 'label': 'Memory Usage', 'unit': '%', 'precision': 1},
        },
        'GPU Performance': {
            'Gpu Usage': {'icon': '🚀', 'label': 'GPU Usage', 'unit': '%', 'precision': 1},
            'Gpu Memory': {'icon': '📊', 'label': 'GPU Memory %', 'unit': '%', 'precision': 1},
            'Gpu Memory Gb': {'icon': '💿', 'label': 'GPU Memory', 'unit': 'GB', 'precision': 2},
        },
        'Inference Metrics': {
            'Inference Latency': {'icon': '⏱️', 'label': 'Latency', 'unit': 'ms', 'precision': 1},
            'Throughput': {'icon': '🔄', 'label': 'Throughput', 'unit': 'tokens/s', 'precision': 2},
            'Total Inference Count': {'icon': '📈', 'label': 'Total Inferences', 'unit': '', 'precision': 0},
        },
        'System Info': {
            'Device': {'icon': '🔧', 'label': 'Device', 'unit': '', 'precision': 0},
        }
    }
    
    # Display metrics in organized groups
    for group_name, group_metrics in metric_groups.items():
        # Check if any metrics in this group exist
        group_has_data = any(metric_key in metrics for metric_key in group_metrics.keys())
        
        if group_has_data:
            st.markdown(f"**{group_name}**")
            
            # Create columns for this group
            cols = st.columns(min(len([k for k in group_metrics.keys() if k in metrics]), 3))
            col_idx = 0
            
            for metric_key, metric_info in group_metrics.items():
                if metric_key in metrics:
                    with cols[col_idx % len(cols)]:
                        raw_value = metrics[metric_key]
                        formatted_value = format_metric_value_simple(raw_value, metric_info)
                        color = get_metric_color_simple(metric_key, raw_value)
                        
                        # Create a metric card
                        st.markdown(f"""
                        <div style='background-color: white; padding: 0.75rem; border-radius: 0.5rem; 
                                    border-left: 4px solid {color}; margin-bottom: 0.5rem; 
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <div style='display: flex; align-items: center; justify-content: space-between;'>
                                <div>
                                    <div style='font-size: 0.8rem; color: #666; margin-bottom: 0.2rem;'>
                                        {metric_info['icon']} {metric_info['label']}
                                    </div>
                                    <div style='font-size: 1.1rem; font-weight: bold; color: {color};'>
                                        {formatted_value}
                                    </div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    col_idx += 1

def display_rag_response_enhanced(answer, query, k_chunks, max_tokens, temperature, processing_time=None):
    """Display RAG response with enhanced formatting and visual appeal"""
    
    # Create response options
    display_option = st.radio(
        "Choose response display format:",
        ["Professional", "Detailed", "Simple"],
        horizontal=True,
        key="response_display_format",
        index=0
    )
    
    if display_option == "Professional":
        display_professional_response(answer, query, k_chunks, max_tokens, temperature, processing_time)
    elif display_option == "Detailed":
        display_detailed_response(answer, query, k_chunks, max_tokens, temperature, processing_time)
    else:
        display_simple_response(answer, query)

def display_professional_response(answer, query, k_chunks, max_tokens, temperature, processing_time=None):
    """Display response in a professional, clean format"""
    
    # Main response card
    st.markdown("""
    <div style='background-color: white; padding: 1.5rem; border-radius: 1rem; margin: 1rem 0; 
                box-shadow: 0 4px 12px rgba(0,0,0,0.1); border: 1px solid #dee2e6;'>
        <div style='padding: 1.5rem; border-radius: 0.75rem;'>
            <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
                <span style='font-size: 1.5rem; margin-right: 0.5rem;'>🤖</span>
                <h3 style='color: #2c3e50; margin: 0; font-size: 1.2rem; font-weight: 600;'>
                    AI Assistant Response
                </h3>
            </div>
    """, unsafe_allow_html=True)
    
    # Display the answer with proper formatting
    st.markdown(f"""
    <div style='font-size: 1rem; line-height: 1.6; color: #ffffff; 
                text-align: left; padding: 0.5rem 0;'>
        {format_answer_text(answer)}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Query details in an elegant expandable section
    with st.expander("📋 Query Details & Metadata", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Query Information**")
            st.info(f"🔍 **Question:** {query}")
            st.info(f"📄 **Context Chunks:** {k_chunks}")
            if processing_time:
                st.info(f"⏱️ **Processing Time:** {processing_time:.1f}s")
        
        with col2:
            st.markdown("**Generation Parameters**")
            st.info(f"📝 **Max Tokens:** {max_tokens}")
            st.info(f"🌡️ **Temperature:** {temperature}")
            st.info(f"📊 **Response Length:** {len(answer.split())} words")

def display_detailed_response(answer, query, k_chunks, max_tokens, temperature, processing_time=None):
    """Display response with detailed breakdown and analysis"""
    
    # Header with gradient
    st.markdown("""
    <div style='background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%); 
                padding: 1rem; border-radius: 0.75rem; text-align: center; margin: 1rem 0;'>
        <h2 style='color: #2c3e50; margin: 0; font-size: 1.3rem; font-weight: 700;'>
            🔍 Detailed RAG Analysis
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📝 Response", "📊 Analysis", "⚙️ Settings"])
    
    with tab1:
        # Main response
        st.markdown("""
        <div style='background-color: white; padding: 1.5rem; border-radius: 0.5rem; 
                    border: 1px solid #dee2e6; margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
            <h4 style='color: #2c3e50; margin-bottom: 1rem; font-size: 1.1rem;'>
                🤖 AI Generated Response
            </h4>
        """, unsafe_allow_html=True)
        
        # Format and display answer
        formatted_answer = format_answer_text(answer)
        st.markdown(f"""
        <div style='font-size: 1rem; line-height: 1.6; color: #2c3e50;'>
            {formatted_answer}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        # Response analysis
        word_count = len(answer.split())
        char_count = len(answer)
        sentence_count = len([s for s in answer.split('.') if s.strip()])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📝 Words", word_count)
        with col2:
            st.metric("📄 Characters", char_count)
        with col3:
            st.metric("📖 Sentences", sentence_count)
        with col4:
            if processing_time:
                st.metric("⚡ Speed", f"{word_count/processing_time:.1f} w/s" if processing_time > 0 else "N/A")
        
        # Quality indicators
        st.markdown("**Response Quality Indicators**")
        quality_score = calculate_response_quality(answer)
        
        col1, col2 = st.columns(2)
        with col1:
            st.progress(quality_score['completeness'], text=f"Completeness: {quality_score['completeness']:.0%}")
            st.progress(quality_score['coherence'], text=f"Coherence: {quality_score['coherence']:.0%}")
        with col2:
            st.progress(quality_score['relevance'], text=f"Relevance: {quality_score['relevance']:.0%}")
            st.progress(quality_score['informativeness'], text=f"Informativeness: {quality_score['informativeness']:.0%}")
    
    with tab3:
        # Settings used
        st.markdown("**Query Configuration**")
        
        settings_data = {
            "Parameter": ["Query", "Context Chunks", "Max Tokens", "Temperature", "Processing Time"],
            "Value": [
                query[:50] + "..." if len(query) > 50 else query,
                str(k_chunks),
                str(max_tokens),
                str(temperature),
                f"{processing_time:.2f}s" if processing_time else "N/A"
            ]
        }
        
        import pandas as pd
        settings_df = pd.DataFrame(settings_data)
        st.dataframe(settings_df, use_container_width=True, hide_index=True)

def display_simple_response(answer, query):
    """Display response in a clean, simple format"""
    
    st.markdown("""
    <div style='background-color: white; padding: 1.25rem; border-radius: 0.75rem; 
                border: 1px solid #dee2e6; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
        <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
            <span style='font-size: 1.2rem; margin-right: 0.5rem;'>🤖</span>
            <h4 style='color: #2c3e50; margin: 0; font-size: 1.1rem;'>AI Response</h4>
        </div>
    """, unsafe_allow_html=True)
    
    # Clean, readable answer
    formatted_answer = format_answer_text(answer)
    st.markdown(f"""
    <div style='font-size: 1rem; line-height: 1.7; color: #2c3e50; text-align: justify;'>
        {formatted_answer}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def format_answer_text(answer):
    """Format the answer text for better readability"""
    # Clean up the answer
    cleaned_answer = answer.strip()
    
    # Handle common formatting issues
    if cleaned_answer.startswith("Answer:"):
        cleaned_answer = cleaned_answer.replace("Answer:", "", 1).strip()
    
    # Remove excessive repetition (common with quantized models)
    words = cleaned_answer.split()
    if len(words) > 3:
        # Check for repetitive patterns
        if len(set(words[:10])) <= 2:  # If first 10 words have 2 or fewer unique words
            cleaned_answer = "⚠️ The model appears to be generating repetitive output. This may indicate an issue with model quantization or prompt formatting."
            return f"<div style='color: #856404; background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; border: 1px solid #ffc107;'>{cleaned_answer}</div>"
    
    # Split into paragraphs for better readability
    paragraphs = cleaned_answer.split('\n\n')
    formatted_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if para:
            # Add proper spacing and formatting
            formatted_paragraphs.append(f"<p style='margin-bottom: 1rem;'>{para}</p>")
    
    return ''.join(formatted_paragraphs) if formatted_paragraphs else cleaned_answer

def calculate_response_quality(answer):
    """Calculate basic quality metrics for the response"""
    words = answer.split()
    
    # Simple heuristics for quality assessment
    quality_metrics = {
        'completeness': min(len(words) / 50, 1.0),  # Based on response length
        'coherence': 1.0 - (len(set(words[:10])) <= 2) * 0.8,  # Penalize repetition
        'relevance': min(len([w for w in words if len(w) > 3]) / max(len(words), 1), 1.0),  # Ratio of meaningful words
        'informativeness': min(len(set(words)) / max(len(words), 1), 1.0)  # Vocabulary diversity
    }
    
    return quality_metrics

def format_metric_value_simple(raw_value, metric_info):
    """Format metric values with appropriate units and precision"""
    try:
        # Handle non-numeric values (like 'CPU' for device)
        if not isinstance(raw_value, (int, float)):
            try:
                numeric_value = float(raw_value)
            except (ValueError, TypeError):
                return str(raw_value)  # Return as-is for non-numeric values
        else:
            numeric_value = raw_value
        
        # Apply precision formatting
        if metric_info['precision'] == 0:
            formatted = f"{int(numeric_value)}"
        else:
            formatted = f"{numeric_value:.{metric_info['precision']}f}"
        
        # Add unit if specified
        if metric_info['unit']:
            return f"{formatted} {metric_info['unit']}"
        else:
            return formatted
            
    except Exception:
        return str(raw_value)

def get_metric_color_simple(metric, value):
    """Determine appropriate color based on metric type and value"""
    try:
        numeric_value = float(value) if isinstance(value, str) else value
        
        # Color coding based on performance implications
        if metric in ['Cpu Usage', 'Memory Usage', 'Gpu Usage', 'Gpu Memory']:
            if numeric_value > 80:
                return '#dc3545'  # Red for high usage
            elif numeric_value > 60:
                return '#ffc107'  # Yellow for medium usage
            else:
                return '#28a745'  # Green for low usage
        elif metric == 'Inference Latency':
            if numeric_value > 1000:  # More than 1 second
                return '#dc3545'  # Red for high latency
            elif numeric_value > 500:  # More than 500ms
                return '#ffc107'  # Yellow for medium latency
            else:
                return '#28a745'  # Green for low latency
        elif metric == 'Throughput':
            if numeric_value > 10:
                return '#28a745'  # Green for high throughput
            elif numeric_value > 5:
                return '#ffc107'  # Yellow for medium throughput
            else:
                return '#dc3545'  # Red for low throughput
        else:
            return '#17a2b8'  # Default blue color
            
    except (ValueError, TypeError):
        return '#6c757d'  # Gray for non-numeric values

def display_performance_grid(metrics):
    """Display performance metrics in a grid layout using Streamlit metrics"""
    if not metrics:
        st.warning("No performance data available")
        return
    
    st.markdown("### 📊 Performance Metrics")
    
    # Create a 3-column grid
    cols = st.columns(3)
    
    metric_info = {
        'Cpu Usage': {'icon': '🖥️', 'label': 'CPU', 'unit': '%'},
        'Memory Usage': {'icon': '💾', 'label': 'Memory', 'unit': '%'},
        'Gpu Usage': {'icon': '🚀', 'label': 'GPU', 'unit': '%'},
        'Gpu Memory Gb': {'icon': '💿', 'label': 'GPU Memory', 'unit': 'GB'},
        'Inference Latency': {'icon': '⏱️', 'label': 'Latency', 'unit': 'ms'},
        'Throughput': {'icon': '🔄', 'label': 'Throughput', 'unit': 'tok/s'},
        'Total Inference Count': {'icon': '📈', 'label': 'Inferences', 'unit': ''},
        'Device': {'icon': '🔧', 'label': 'Device', 'unit': ''}
    }
    
    col_idx = 0
    for metric_key, value in metrics.items():
        if metric_key in metric_info:
            info = metric_info[metric_key]
            
            with cols[col_idx % 3]:
                # Format value
                try:
                    if metric_key == 'Device':
                        display_value = str(value)
                    elif metric_key == 'Total Inference Count':
                        display_value = str(int(float(value)))
                    elif metric_key in ['Gpu Memory Gb', 'Throughput']:
                        display_value = f"{float(value):.2f}"
                    else:
                        display_value = f"{float(value):.1f}"
                except:
                    display_value = str(value)
                
                # Add unit if not device
                if info['unit'] and metric_key != 'Device':
                    display_value += f" {info['unit']}"
                
                # Create metric box
                st.metric(
                    label=f"{info['icon']} {info['label']}",
                    value=display_value
                )

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Document Question & Answering</p>', unsafe_allow_html=True)
    
    # Initialize session state for query
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    if 'process_query' not in st.session_state:
        st.session_state.process_query = False
    if 'performance_result' not in st.session_state:
        st.session_state.performance_result = None
    if 'show_performance' not in st.session_state:
        st.session_state.show_performance = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 0.75rem; border-radius: 0.5rem; border: 1px solid #dee2e6; margin-bottom: 0.75rem;'>
            <h3 style='color: #1f77b4; margin-bottom: 0.75rem; text-align: center; font-size: 1rem;'>⚙️ Configuration</h3>
        """, unsafe_allow_html=True)
        
        # System status with better styling
        st.markdown("**🔧 System Status**")
        system_info = get_system_info()
        
        if system_info["model_available"]:
            st.success("✅ Model Loaded")
        else:
            st.error("❌ Model Not Found")
            
        if system_info["vector_store_available"]:
            st.success("✅ Vector Store Ready")
        else:
            st.error("❌ Vector Store Not Found")
            
        if system_info["gpu_available"]:
            st.success(f"✅ GPU Available: {system_info['gpu_name']}")
            st.info(f"GPU Memory: {system_info['gpu_memory']:.1f} GB")
        else:
            st.warning("⚠️ GPU Not Available (Using CPU)")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Query parameters with better styling
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 0.75rem; border-radius: 0.5rem; border: 1px solid #dee2e6; margin-bottom: 0.75rem;'>
            <h3 style='color: #1f77b4; margin-bottom: 0.75rem; text-align: center; font-size: 1rem;'>🎛️ Query Settings</h3>
        """, unsafe_allow_html=True)
        
        k_chunks = st.slider("📄 Number of chunks (k)", 1, 10, 5, help="Number of document chunks to retrieve")
        max_tokens = st.slider("📝 Max tokens", 50, 500, 200, help="Maximum length of the generated response")
        temperature = st.slider("🌡️ Temperature", 0.0, 2.0, 0.7, 0.1, help="Controls response creativity (higher = more creative)")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Performance monitoring with better styling
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 0.75rem; border-radius: 0.5rem; border: 1px solid #dee2e6;'>
            <h3 style='color: #1f77b4; margin-bottom: 0.75rem; text-align: center; font-size: 1rem;'>📊 Performance</h3>
        """, unsafe_allow_html=True)
        
        if st.button("📊 Run Performance Test", use_container_width=True):
            st.session_state.show_performance = True
            st.session_state.performance_result = None
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Main Query Section with better styling
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border: 1px solid #dee2e6; margin-bottom: 0.75rem;'>
            <h3 style='color: #1f77b4; margin-bottom: 0.75rem; text-align: center; font-size: 1.2rem;'>💬 Ask a Question</h3>
        """, unsafe_allow_html=True)
        
        # Query input with better styling
        query = st.text_area(
            "Enter your question about the documents:",
            value=st.session_state.current_query,
            placeholder="e.g., What is Procyon and what are its main features?",
            height=80,
            help="Type your question here and the AI will search through the documents to find relevant answers."
        )
        
        # Submit button with better styling
        if st.button("🚀 Submit Query", type="primary", use_container_width=True, help="Click to process your query"):
            if query.strip():
                st.session_state.process_query = True
                st.session_state.current_query = query
                st.rerun()
            else:
                st.warning("Please enter a question")
        
        # Process query if triggered
        if st.session_state.process_query and st.session_state.current_query:
            answer, error, processing_time = process_query(st.session_state.current_query, k_chunks, max_tokens, temperature)
            
            if answer:
                # Display the enhanced response
                display_rag_response_enhanced(
                    answer, 
                    st.session_state.current_query, 
                    k_chunks, 
                    max_tokens, 
                    temperature, 
                    processing_time
                )
            else:
                st.error(f"Error: {error}")
            
            # Reset the process flag
            st.session_state.process_query = False
        
        # Process performance test if triggered
        if st.session_state.show_performance:
            with st.spinner("Running performance test..."):
                try:
                    result = subprocess.run(
                        ["python", "rag_cli.py", "performance", "--duration", "30"],
                        capture_output=True,
                        text=True,
                        cwd=Path(__file__).parent
                    )
                    if result.returncode == 0:
                        st.session_state.performance_result = result.stdout
                    else:
                        st.session_state.performance_result = f"Performance test failed:\n{result.stderr}"
                except Exception as e:
                    st.session_state.performance_result = f"Error: {e}"
            
            # Display performance results
            if st.session_state.performance_result:
                metrics = parse_performance_output(st.session_state.performance_result)
                if metrics:
                    # Offer different presentation options
                    display_option = st.radio(
                        "Choose display format:",
                        ["Cards", "Grid", "Raw Output"],
                        horizontal=True,
                        key="display_format",
                        index=0
                    )
                    
                    if display_option == "Cards":
                        display_performance_metrics(metrics)
                    elif display_option == "Grid":
                        display_performance_grid(metrics)
                    else:
                        # Show raw output in an expandable section
                        with st.expander("📊 Raw Performance Output", expanded=True):
                            st.text(st.session_state.performance_result)
                else:
                    # Fallback to raw output if parsing fails
                    st.markdown("""
                    <div style='background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; border: 1px solid #ffc107; margin: 0.5rem 0;'>
                        <h4 style='color: #856404; margin-bottom: 0.5rem; font-size: 1rem;'>📊 Performance Test Results</h4>
                        <div style='font-size: 14px; line-height: 1.4;'>
                    """, unsafe_allow_html=True)
                    st.text(st.session_state.performance_result)
                    st.markdown('</div></div>', unsafe_allow_html=True)
            
            # Reset the performance flag
            st.session_state.show_performance = False
    
    with col2:
        # Quick Stats Section with compact styling
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 0.75rem; border-radius: 0.5rem; border: 1px solid #dee2e6; margin-bottom: 0.75rem;'>
            <h3 style='color: #1f77b4; margin-bottom: 0.5rem; text-align: center; font-size: 0.9rem;'>📈 System Status</h3>
        """, unsafe_allow_html=True)
        
        # System metrics in a compact grid layout
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            # Model Status
            model_status = "✅ Ready" if system_info["model_available"] else "❌ Missing"
            st.markdown(f"""
            <div style='background-color: white; padding: 0.5rem; border-radius: 0.25rem; margin-bottom: 0.25rem; border-left: 3px solid #28a745; font-size: 0.8rem;'>
                <div style='font-weight: bold; color: #495057;'>🤖 Model</div>
                <div style='color: #28a745; font-size: 0.75rem;'>{model_status}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Vector Store Status
            vector_status = "✅ Ready" if system_info["vector_store_available"] else "❌ Missing"
            st.markdown(f"""
            <div style='background-color: white; padding: 0.5rem; border-radius: 0.25rem; margin-bottom: 0.25rem; border-left: 3px solid #17a2b8; font-size: 0.8rem;'>
                <div style='font-weight: bold; color: #495057;'>🗄️ Vector Store</div>
                <div style='color: #17a2b8; font-size: 0.75rem;'>{vector_status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_col2:
            # GPU Status
            gpu_status = "✅ Active" if system_info["gpu_available"] else "⚠️ CPU"
            gpu_color = "#28a745" if system_info["gpu_available"] else "#ffc107"
            st.markdown(f"""
            <div style='background-color: white; padding: 0.5rem; border-radius: 0.25rem; margin-bottom: 0.25rem; border-left: 3px solid {gpu_color}; font-size: 0.8rem;'>
                <div style='font-weight: bold; color: #495057;'>🚀 GPU</div>
                <div style='color: {gpu_color}; font-size: 0.75rem;'>{gpu_status}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # GPU Memory
            if system_info["gpu_available"]:
                gpu_memory = f"{system_info['gpu_memory']:.1f} GB"
                st.markdown(f"""
                <div style='background-color: white; padding: 0.5rem; border-radius: 0.25rem; margin-bottom: 0.25rem; border-left: 3px solid #6f42c1; font-size: 0.8rem;'>
                    <div style='font-weight: bold; color: #495057;'>💾 Memory</div>
                    <div style='color: #6f42c1; font-size: 0.75rem;'>{gpu_memory}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Sample Queries Section with compact styling
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 0.75rem; border-radius: 0.5rem; border: 1px solid #dee2e6;'>
            <h3 style='color: #1f77b4; margin-bottom: 0.5rem; text-align: center; font-size: 0.9rem;'>💡 Quick Queries</h3>
        """, unsafe_allow_html=True)
        
        sample_queries = [
            "What is Procyon and what are its main features?",
            "How does the benchmark system work?",
            "What are the different types of tests available?",
            "How can I run Procyon benchmarks?",
            "What industries does Procyon serve?"
        ]
        
        for i, sample_query in enumerate(sample_queries):
            if st.button(f"🔍 {sample_query[:25]}...", key=f"sample_{i}", use_container_width=True):
                # Set the query in session state and trigger processing
                st.session_state.current_query = sample_query
                st.session_state.process_query = True
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer with better styling
    st.markdown("---")
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border: 1px solid #dee2e6; text-align: center; color: #666; font-size: 0.8rem;'>
        <p style='margin-bottom: 0.25rem;'><strong>Built with Streamlit</strong> • Powered by OpenVINO GenAI • GPU Accelerated</p>
        <p style='margin-bottom: 0;'>RAG System v1.0 • Llama-3.1-8B-Instruct • FAISS Vector Search</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()