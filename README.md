# 🤖 RAG-based Q&A System for ITMO Master's Programs

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> A Retrieval-Augmented Generation (RAG) system that answers questions about ITMO University's Master's programs in Artificial Intelligence and AI Product using parsed website data.

## 📌 Overview

This repository contains a question-answering system that leverages RAG architecture to provide accurate information about ITMO University's Master's programs. The system retrieves relevant information from parsed program descriptions and generates natural language answers using a pre-trained language model.

## 🚀 Features

- 🔍 **Smart Retrieval**: Uses TF-IDF vectorization for efficient document retrieval
- 🧠 **AI-Powered Generation**: Employs transformer models for natural language responses
- 📚 **Comprehensive Knowledge Base**: Contains detailed information about:
  - Program descriptions
  - Admission requirements
  - Career opportunities
  - Partnerships and collaborations
  - Scholarship information
- 💬 **Interactive Q&A**: Command-line interface for real-time questioning

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/itmo-rag-qa.git
cd itmo-rag-qa
```

Install required packages:
```bash
pip install -r requirements.txt
```
Or install manually:
```bash
pip install requests beautifulsoup4 scikit-learn transformers torch
```
🛠️ Usage
```bash
python main.py
```
Example Questions
"Какие стипендии доступны для магистрантов?"
"Какие компании являются партнерами программы по AI?"
"Какие существуют варианты выпускной работы?"
"Какие направления подготовки есть в программе AI?"

🏗️ Project Structure
```
├── main.py           # Main RAG Q&A system
├── requirements.txt    # Python dependencies
└── README.md          # This file
```
🧪 How It Works

🔧 Data Processing: Program information is parsed and preprocessed into searchable chunks

🔍 Retrieval: User questions are matched against the knowledge base using TF-IDF similarity

🧠 Generation: Relevant text fragments are fed to a transformer model to generate natural answers

💬 Response: Generated answers are displayed to the user in real-time

📊 Models Used
Retrieval: TF-IDF vectorization with cosine similarity
Generation: google/mt5-small transformer model (multilingual, supports Russian)

⚠️ Important Notes
First Run: Model download may take several minutes on first execution
System Requirements: At least 4GB RAM recommended (8GB+ for better performance)
GPU Support: Automatically utilizes GPU if CUDA is available

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Thanks to ITMO University for providing detailed program information
Built with 🤗 Hugging Face transformers library
Uses scikit-learn for efficient text processing

📞 Support
For questions and support, please open an issue in the GitHub repository.
