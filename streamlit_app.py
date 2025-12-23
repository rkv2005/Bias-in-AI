"""
Interactive Gender Bias Detection & Mitigation
Real-time text analysis and counterfactual generation
"""

import streamlit as st
import pandas as pd
import re

# Page config
st.set_page_config(
    page_title="Gender Bias Mitigation",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .subheader {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .text-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .metric-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

class BiasMitigation:
    """Gender bias detection and mitigation"""
    
    @staticmethod
    def create_counterfactual(text):
        """Create gender-swapped version of text"""
        swap_map = {
            'he': 'she', 'she': 'he',
            'him': 'her', 'her': 'him',
            'his': 'her', 'hers': 'his',
            'himself': 'herself', 'herself': 'himself',
            'man': 'woman', 'woman': 'man',
            'men': 'women', 'women': 'men',
            'male': 'female', 'female': 'male',
            'boy': 'girl', 'girl': 'boy',
            'boys': 'girls', 'girls': 'boys',
            'father': 'mother', 'mother': 'father',
            'son': 'daughter', 'daughter': 'son',
            'brother': 'sister', 'sister': 'brother',
            'uncle': 'aunt', 'aunt': 'uncle',
            'nephew': 'niece', 'niece': 'nephew',
            'husband': 'wife', 'wife': 'husband',
            'boyfriend': 'girlfriend', 'girlfriend': 'boyfriend',
            'guy': 'gal', 'gal': 'guy',
            'gentleman': 'lady', 'lady': 'gentleman',
            'sir': 'madam', 'madam': 'sir',
            'mr': 'ms', 'ms': 'mr',
        }
        
        words = text.split()
        swapped_words = []
        
        for word in words:
            # Preserve original word
            original_word = word
            # Remove punctuation for matching
            clean_word = word.strip('.,!?;:"\'-').lower()
            
            if clean_word in swap_map:
                swapped = swap_map[clean_word]
                
                # Preserve capitalization
                if word[0].isupper():
                    swapped = swapped.capitalize()
                if word.isupper():
                    swapped = swapped.upper()
                
                # Reattach punctuation
                for char in word:
                    if char in '.,!?;:"\'-' and char not in swapped:
                        if word.index(char) < len(word) - 1:
                            # Punctuation in middle (like "don't")
                            continue
                        swapped += char
                
                swapped_words.append(swapped)
            else:
                swapped_words.append(original_word)
        
        return ' '.join(swapped_words)
    
    @staticmethod
    def analyze_pronouns(text):
        """Count gendered pronouns in text"""
        text_lower = text.lower()
        
        male_pronouns = ['he', 'him', 'his', 'himself']
        female_pronouns = ['she', 'her', 'hers', 'herself']
        
        male_count = sum([len(re.findall(r'\b' + p + r'\b', text_lower)) for p in male_pronouns])
        female_count = sum([len(re.findall(r'\b' + p + r'\b', text_lower)) for p in female_pronouns])
        
        return male_count, female_count
    
    @staticmethod
    def detect_gendered_words(text):
        """Detect all gendered words in text"""
        text_lower = text.lower()
        
        gendered_words = {
            'Male': ['he', 'him', 'his', 'himself', 'man', 'men', 'male', 'boy', 'boys',
                    'father', 'son', 'brother', 'uncle', 'nephew', 'husband', 'boyfriend',
                    'gentleman', 'guy', 'sir', 'mr'],
            'Female': ['she', 'her', 'hers', 'herself', 'woman', 'women', 'female', 'girl', 'girls',
                      'mother', 'daughter', 'sister', 'aunt', 'niece', 'wife', 'girlfriend',
                      'lady', 'gal', 'madam', 'ms', 'mrs']
        }
        
        found_words = {'Male': [], 'Female': []}
        
        for gender, words in gendered_words.items():
            for word in words:
                matches = re.findall(r'\b' + word + r'\b', text_lower)
                if matches:
                    found_words[gender].extend(matches)
        
        return found_words

# Initialize
mitigation = BiasMitigation()

# Header
st.markdown('<p class="main-header">⚖️ Gender Bias Mitigation Tool</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Analyze and neutralize gender bias in text with counterfactual generation</p>', unsafe_allow_html=True)
st.markdown("---")

# Main content
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📝 Input Text")
    
    # Example selector
    example = st.selectbox(
        "Or select an example:",
        [
            "Custom input",
            "The engineer designed the system and he tested it thoroughly.",
            "The nurse administered the medication and she checked the patient's vital signs.",
            "The CEO announced that he would be stepping down next quarter.",
            "She is an excellent programmer with strong technical skills.",
            "The pilot completed his training and received his certification.",
            "The teacher explained the lesson and her students understood it well.",
            "The doctor prescribed medicine because he believed it would help.",
            "The secretary organized the files and she scheduled all the meetings."
        ]
    )
    
    # Text input
    if example == "Custom input":
        user_text = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Type or paste your text here..."
        )
    else:
        user_text = st.text_area(
            "Enter text to analyze:",
            value=example,
            height=150
        )
    
    analyze_button = st.button("🔍 Analyze & Generate Counterfactual", type="primary", use_container_width=True)

with col_right:
    st.subheader("⚡ Counterfactual Output")
    
    if user_text and analyze_button:
        # Generate counterfactual
        counterfactual_text = mitigation.create_counterfactual(user_text)
        
        # Display counterfactual
        st.markdown('<div class="text-box">', unsafe_allow_html=True)
        st.write(counterfactual_text)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Copy button
        st.code(counterfactual_text, language=None)
    else:
        st.info("👈 Enter text and click 'Analyze' to see the gender-swapped version")

# Analysis section
if user_text and analyze_button:
    st.markdown("---")
    st.subheader("📊 Bias Analysis")
    
    # Pronoun counts
    male_count, female_count = mitigation.analyze_pronouns(user_text)
    total_pronouns = male_count + female_count
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Male Pronouns", male_count)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Female Pronouns", female_count)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Total Gendered", total_pronouns)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        if total_pronouns > 0:
            bias_score = (male_count - female_count) / total_pronouns
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Bias Score", f"{bias_score:+.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Bias Score", "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Gendered words detection
    st.markdown("---")
    st.subheader("🔍 Detected Gendered Terms")
    
    found_words = mitigation.detect_gendered_words(user_text)
    
    col_male, col_female = st.columns(2)
    
    with col_male:
        st.markdown("**Male-associated words:**")
        if found_words['Male']:
            st.write(", ".join(set(found_words['Male'])))
        else:
            st.write("_None detected_")
    
    with col_female:
        st.markdown("**Female-associated words:**")
        if found_words['Female']:
            st.write(", ".join(set(found_words['Female'])))
        else:
            st.write("_None detected_")
    
    # Interpretation
    st.markdown("---")
    st.subheader("💡 Interpretation")
    
    if total_pronouns == 0:
        st.info("ℹ️ No gendered pronouns detected in the text.")
    elif male_count > female_count * 2:
        st.warning(f"⚠️ **Male-biased**: Text contains significantly more male pronouns ({male_count}) than female ({female_count}).")
    elif female_count > male_count * 2:
        st.warning(f"⚠️ **Female-biased**: Text contains significantly more female pronouns ({female_count}) than male ({male_count}).")
    else:
        st.success(f"✓ **Relatively balanced**: Text has similar male ({male_count}) and female ({female_count}) pronoun usage.")
    
    # Mitigation explanation
    st.markdown("---")
    st.subheader("🛠️ How Counterfactual Augmentation Works")
    
    st.markdown("""
    **Counterfactual Data Augmentation** is a bias mitigation technique that:
    
    1. **Identifies** gendered words in the text (pronouns, nouns, titles)
    2. **Swaps** them with opposite-gender equivalents (he→she, man→woman, etc.)
    3. **Preserves** capitalization, punctuation, and sentence structure
    4. **Generates** a parallel version that can be used to:
       - Balance training datasets
       - Test model robustness
       - Identify gender-dependent model behaviors
    
    **Use cases:**
    - 📚 Augment training data to reduce gender bias in ML models
    - 🧪 Test if your model makes different predictions for different genders
    - 📊 Analyze corpus-level gender distribution in text datasets
    """)

# Sidebar
st.sidebar.title("About")
st.sidebar.info("""
**Gender Bias Mitigation Tool**

This tool helps detect and mitigate gender bias in text through counterfactual generation.

**Features:**
- Real-time text analysis
- Gender pronoun counting
- Counterfactual text generation
- Bias score calculation
- Gendered term detection

**Methodology:**
Based on counterfactual data augmentation techniques used in fairness research.
""")

st.sidebar.markdown("---")
st.sidebar.subheader("📖 Example Use Cases")
st.sidebar.markdown("""
- Analyze job descriptions for bias
- Balance training datasets
- Test model fairness
- Audit content for gender neutrality
- Generate parallel test cases
""")

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Bias Score")
st.sidebar.markdown("""
- **+1.0**: Only male pronouns
- **0.0**: Balanced usage
- **-1.0**: Only female pronouns
- **> +0.5**: Male-biased
- **< -0.5**: Female-biased
""")
