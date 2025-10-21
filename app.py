import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import time

# Import our FTQC modules
from ftqc_algorithms import (
    QuantumPhaseEstimation,
    QuantumResourceEstimator,
    PhotonicCircuitOptimizer,
    CVQuantumAlgorithms
)
from hardware_backends import (
    PsiQuantumBackend,
    XanaduBackend,
    GenericFTQCBackend
)

# Page configuration
st.set_page_config(
    page_title="Orquestra FTQC Platform",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .hardware-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">⚛️ Orquestra FTQC Platform</h1>', unsafe_allow_html=True)
    st.markdown("**Fault-Tolerant Quantum Computing Algorithms for Next-Generation Hardware**")
    
    # Sidebar for hardware selection and parameters
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Hardware platform selection
        hardware_platform = st.selectbox(
            "Select Hardware Platform",
            ["PsiQuantum", "Xanadu", "Generic FTQC", "Compare All"],
            help="Choose the target quantum hardware platform"
        )
        
        # Algorithm selection
        algorithm_type = st.selectbox(
            "Select Algorithm",
            ["Quantum Phase Estimation", "Resource Estimation", "Circuit Optimization", "CV Algorithms"],
            help="Choose the quantum algorithm to run"
        )
        
        st.divider()
        
        # Algorithm-specific parameters
        if algorithm_type == "Quantum Phase Estimation":
            st.subheader("QPE Parameters")
            num_qubits = st.slider("Number of Qubits", 4, 20, 8)
            precision_bits = st.slider("Precision Bits", 4, 16, 8)
            eigenvalue = st.number_input("Target Eigenvalue", 0.0, 1.0, 0.25, step=0.01)
            
        elif algorithm_type == "Resource Estimation":
            st.subheader("Resource Parameters")
            problem_size = st.slider("Problem Size", 10, 1000, 100)
            error_budget = st.selectbox("Error Budget", ["10^-3", "10^-6", "10^-9"])
            logical_error_rate = st.number_input("Logical Error Rate", 1e-15, 1e-3, 1e-6, format="%.2e")
            
        elif algorithm_type == "Circuit Optimization":
            st.subheader("Optimization Parameters")
            circuit_depth = st.slider("Circuit Depth", 10, 200, 50)
            optimization_level = st.selectbox("Optimization Level", ["Basic", "Intermediate", "Advanced"])
            target_fidelity = st.slider("Target Fidelity", 0.9, 0.999, 0.99, step=0.001)
            
        elif algorithm_type == "CV Algorithms":
            st.subheader("CV Parameters")
            num_modes = st.slider("Number of Modes", 2, 10, 4)
            squeezing_param = st.slider("Squeezing Parameter", 0.0, 2.0, 1.0, step=0.1)
            cutoff_dim = st.slider("Cutoff Dimension", 10, 50, 20)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"🚀 {algorithm_type} on {hardware_platform}")
        
        # Run algorithm button
        if st.button("Run Algorithm", type="primary", use_container_width=True):
            with st.spinner(f"Running {algorithm_type} on {hardware_platform}..."):
                results = run_algorithm(
                    algorithm_type, hardware_platform, 
                    locals()  # Pass all local variables as parameters
                )
                display_results(results, algorithm_type, hardware_platform)
    
    with col2:
        st.header("📊 Hardware Info")
        display_hardware_info(hardware_platform)
    
    # Additional sections
    st.divider()
    
    # Comparison section
    if hardware_platform == "Compare All":
        st.header("🔍 Hardware Comparison")
        display_hardware_comparison(algorithm_type)
    
    # Performance metrics
    st.header("📈 Performance Metrics")
    display_performance_dashboard()

def run_algorithm(algorithm_type: str, hardware_platform: str, params: Dict) -> Dict:
    """Run the selected algorithm on the chosen hardware platform."""
    
    # Initialize backend
    if hardware_platform == "PsiQuantum":
        backend = PsiQuantumBackend()
    elif hardware_platform == "Xanadu":
        backend = XanaduBackend()
    else:
        backend = GenericFTQCBackend()
    
    # Run algorithm based on type
    if algorithm_type == "Quantum Phase Estimation":
        qpe = QuantumPhaseEstimation(
            num_qubits=params['num_qubits'],
            precision_bits=params['precision_bits']
        )
        results = qpe.run(backend, target_eigenvalue=params['eigenvalue'])
        
    elif algorithm_type == "Resource Estimation":
        estimator = QuantumResourceEstimator()
        results = estimator.estimate_resources(
            problem_size=params['problem_size'],
            error_budget=params['error_budget'],
            backend=backend
        )
        
    elif algorithm_type == "Circuit Optimization":
        optimizer = PhotonicCircuitOptimizer()
        results = optimizer.optimize_circuit(
            depth=params['circuit_depth'],
            optimization_level=params['optimization_level'],
            target_fidelity=params['target_fidelity'],
            backend=backend
        )
        
    elif algorithm_type == "CV Algorithms":
        cv_alg = CVQuantumAlgorithms()
        results = cv_alg.run_gaussian_boson_sampling(
            num_modes=params['num_modes'],
            squeezing=params['squeezing_param'],
            cutoff=params['cutoff_dim'],
            backend=backend
        )
    
    return results

def display_results(results: Dict, algorithm_type: str, hardware_platform: str):
    """Display algorithm results with visualizations."""
    
    st.subheader("🎯 Results")
    
    # Create tabs for different result views
    tab1, tab2, tab3 = st.tabs(["Summary", "Detailed Analysis", "Visualizations"])
    
    with tab1:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Success Probability", f"{results.get('success_prob', 0.95):.3f}")
        with col2:
            st.metric("Execution Time", f"{results.get('exec_time', 1.23):.2f} ms")
        with col3:
            st.metric("Fidelity", f"{results.get('fidelity', 0.99):.4f}")
        with col4:
            st.metric("Resource Cost", f"{results.get('resource_cost', 1000):,}")
    
    with tab2:
        # Detailed results table
        if 'detailed_results' in results:
            st.dataframe(results['detailed_results'], use_container_width=True)
        else:
            st.info("No detailed analysis available for this algorithm.")
    
    with tab3:
        # Visualizations
        if 'plot_data' in results:
            fig = create_result_visualization(results['plot_data'], algorithm_type)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No visualizations available for this algorithm.")

def display_hardware_info(hardware_platform: str):
    """Display information about the selected hardware platform."""
    
    hardware_specs = {
        "PsiQuantum": {
            "Type": "Photonic",
            "Qubits": "1M+ (planned)",
            "Connectivity": "All-to-all",
            "Error Rate": "10^-6",
            "Gate Time": "1 ns",
            "Advantages": ["Room temperature", "Network-ready", "Fault-tolerant"]
        },
        "Xanadu": {
            "Type": "Continuous Variable",
            "Modes": "216",
            "Connectivity": "Programmable",
            "Error Rate": "Variable",
            "Gate Time": "10 ns",
            "Advantages": ["Near-term ready", "Gaussian operations", "Photonic"]
        },
        "Generic FTQC": {
            "Type": "Universal",
            "Qubits": "Configurable",
            "Connectivity": "Configurable",
            "Error Rate": "Configurable",
            "Gate Time": "Configurable",
            "Advantages": ["Flexible", "Comparative analysis", "Benchmarking"]
        }
    }
    
    if hardware_platform in hardware_specs:
        specs = hardware_specs[hardware_platform]
        
        st.markdown(f"### {hardware_platform} Specifications")
        
        for key, value in specs.items():
            if key != "Advantages":
                st.text(f"{key}: {value}")
        
        st.markdown("**Key Advantages:**")
        for advantage in specs["Advantages"]:
            st.text(f"• {advantage}")

def display_hardware_comparison(algorithm_type: str):
    """Display comparison between different hardware platforms."""
    
    # Sample comparison data
    comparison_data = {
        "Platform": ["PsiQuantum", "Xanadu", "Generic FTQC"],
        "Execution Time (ms)": [1.2, 5.8, 2.1],
        "Fidelity": [0.995, 0.987, 0.992],
        "Resource Cost": [1000, 1500, 800],
        "Success Rate": [0.98, 0.94, 0.96]
    }
    
    df = pd.DataFrame(comparison_data)
    
    # Display as table
    st.dataframe(df, use_container_width=True)
    
    # Create comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.bar(df, x="Platform", y="Fidelity", title="Fidelity Comparison")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.bar(df, x="Platform", y="Execution Time (ms)", title="Execution Time Comparison")
        st.plotly_chart(fig2, use_container_width=True)

def display_performance_dashboard():
    """Display overall performance dashboard."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample performance data over time
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        performance_data = {
            'Date': dates,
            'Success Rate': np.random.normal(0.95, 0.02, 30),
            'Average Fidelity': np.random.normal(0.99, 0.005, 30)
        }
        df_perf = pd.DataFrame(performance_data)
        
        fig = px.line(df_perf, x='Date', y=['Success Rate', 'Average Fidelity'], 
                     title="Performance Trends")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Resource utilization
        resource_data = {
            'Resource Type': ['Qubits', 'Gates', 'Memory', 'Time'],
            'Utilization (%)': [75, 82, 68, 91]
        }
        df_resource = pd.DataFrame(resource_data)
        
        fig = px.pie(df_resource, values='Utilization (%)', names='Resource Type',
                    title="Resource Utilization")
        st.plotly_chart(fig, use_container_width=True)

def create_result_visualization(plot_data: Dict, algorithm_type: str) -> go.Figure:
    """Create visualization based on algorithm results."""
    
    if algorithm_type == "Quantum Phase Estimation":
        # Phase estimation probability distribution
        phases = plot_data.get('phases', np.linspace(0, 1, 100))
        probabilities = plot_data.get('probabilities', np.exp(-((phases - 0.25) ** 2) / 0.01))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=phases, y=probabilities, mode='lines', name='Probability'))
        fig.update_layout(title="Phase Estimation Results", xaxis_title="Phase", yaxis_title="Probability")
        
    elif algorithm_type == "Resource Estimation":
        # Resource scaling
        problem_sizes = plot_data.get('sizes', [10, 50, 100, 500, 1000])
        resources = plot_data.get('resources', [100, 2500, 10000, 250000, 1000000])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=problem_sizes, y=resources, mode='lines+markers', name='Resources'))
        fig.update_layout(title="Resource Scaling", xaxis_title="Problem Size", yaxis_title="Resources Required")
        
    else:
        # Default visualization
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Result'))
        fig.update_layout(title="Algorithm Results")
    
    return fig

if __name__ == "__main__":
    main()
