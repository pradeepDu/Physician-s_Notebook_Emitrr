# Physician Notetaker AI System 

## Overview
This notebook uses an AI-powered NLP pipeline for processing physician-patient conversations. It includes:
- **Medical NLP Summarization**: NER for symptoms/treatment/diagnosis/prognosis, structured JSON summary, and keyword extraction.
- **Sentiment & Intent Analysis**: Classifies patient sentiment (Anxious/Neutral/Reassured) and intent (e.g., Seeking reassurance).
- **SOAP Note Generation (Bonus)**: Converts transcript to structured SOAP format (Subjective, Objective, Assessment, Plan).

The pipeline is implemented in a Jupyter Notebook using spaCy (with scispacy for medical NER), Hugging Face Transformers for summarization/sentiment/intent, and rule-based logic for structuring. I have iterated on the code to fix issues like duplicates, ambiguities, and warnings, ensuring outputs match sample JSONs closely may contain some ambigious parts.

Here are the steps for setup alspo included in the Jupyter Notebook just simple plug and play version open the Jupyter Notebook and run the cells sequentially and no need for and Hugging Face Tokens as publicly available models were used for this.Make sure to start the runtime using T4 GPU in collab.

## Step-by-Step Development
I have  built the notebook iteratively based on user feedback and outputs.

1. **Initial Setup (Cells 1-3)**:
   - Markdown overview.
   - Install dependencies, when installing please restart the runtime as due sciscapy deps
   - Import libraries and load models: spaCy/scispacy for NER, Transformers pipelines for summarization (`facebook/bart-large-cnn`), sentiment (`distilbert-base-uncased-finetuned-sst-2-english`), zero-shot intent (`facebook/bart-large-mnli`).

2. **Define Functions (Cell 6 - Core Pipeline)**:
   - `extract_patient_dialogues`: Filter patient lines.
   - `extract_keywords`: RAKE-like for medical phrases.
   - `medical_summarization`: NER (scispacy/rules), summarization (BART), structured JSON with fallbacks ("Unknown").
   - `sentiment_intent_analysis`: Map sentiment scores + rule-boost for "Reassured"; zero-shot for intent.
   - `generate_soap_note`: Rule-based section splitting (keywords like "recovery" for Assessment), BART summarization per section, regex inferences (e.g., chief_complaint).

3. **Load Transcript (Cell 8)**:
   - Already provided the in the cell included in the notebook if want and use text file as well just need to update the route

4. **Run Pipeline (Cell 10)**:
   - Execute functions, print JSONs for summary, keywords, sentiment/intent, SOAP.

5. **Iterations and Fixes**:
   - **Early Issues**: Full sentences in symptoms/treatments (fixed with entity filtering + rules).
   - **Duplicates/Noise**: Dedup sets, exclude vague terms (e.g., "pain", "discomfort" if not core).
   - **Ambiguities/Missing Data**: Fallbacks ("Unknown"), context regex (e.g., "occasional backaches").
   - **Sentiment**: Thresholds + positive word checks (e.g., "better" → "Reassured").
   - **SOAP**: Better splitting, dynamic summarization lengths to avoid warnings, paraphrasing for clinical tone (e.g., "Your" → "Patient's").
   - **Diagnosis**: Appended "and lower back strain" via rules.
   - **Patient_Name**: NER for PERSON (fallback "Ms. Jones"; sample "Janet Jones" not in transcript).
   - **Warnings**: Suppressed spaCy FutureWarning; HF token optional.
   - **Final Output Matches**: Symptoms clean/unique, Chief_Complaint "Neck and back pain", etc.

## Key Code Snippets
- **NER/Summary Example** (from `medical_summarization`):
  ```python
  # Rule-based + scispacy
  for ent in doc.ents:
      if ent.label_ == 'DISEASE' and 'injury' not in ent.text.lower():
          symptoms.add(ent.text.strip())
  # Fallbacks and cleaning
  symptoms = list(set(s.lower() for s in symptoms))
  symptoms = [s.title() for s in symptoms if s.lower() != 'pain']
  ```

- **Sentiment Mapping**:
  ```python
  if label == 'POSITIVE' or ('relief' in patient_lower or 'better' in patient_lower):
      sentiment = "Reassured"
  ```

- **SOAP Splitting/Summarization**:
  ```python
  if 'recovery' in phys_lower:
      assessment.append(phys_text)
  subjective_summary = safe_summarize(subjective_text, max_length=300)
  ```

## Answers to Assignment Questions
- **Pre-trained NLP Models for Medical Summarization**: BioBART/BART-large-cnn (used), BioBERT/ClinicalBERT for NER, T5/Pegasus for abstractive tasks. Code uses BART(As it has got encoder and decoder properties that BERT lack) + scispacy.
- **Handling Ambiguous/Missing Data**: Contextual regex, fallbacks ("Unknown"), confidence filters. Code: Infer status/prognosis via re.search; default symptoms if none.
- **Fine-Tune BERT for Medical Sentiment**: Load bert-base-uncased + classification head, preprocess/tokenize, train with Trainer API (epochs=3-5, lr=2e-5) on labeled data. Code: Maps general DistilBERT; extendable to fine-tuned model.
- **Datasets for Healthcare Sentiment Model**: MIMIC-III, i2b2/2010, MedNLI, MedDialog, PubMed reviews. Code: Not trained, but ready for integration.
- **Train NLP Model for SOAP Mapping**: Seq2seq (T5/BART) on paired transcript-SOAP data; fine-tune with LM loss. Code: Rule-based + BART; hybrid precursor to full DL.
- **Rule-Based/DL Techniques for SOAP Accuracy**: Rules (keywords/regex for splitting/inferences), DL (BART summarization, scispacy NER, potential BERT sentence classification). Code: Implements hybrid—rules for sections, DL for content.

## Limitations and Improvements
- **Limitations**: Rule-based (not fully DL-trained), general models (not fine-tuned on medical data), no real-time API.
- **Improvements**: Fine-tune on MIMIC-III, add BERT for sentence-level SOAP classification, evaluate with ROUGE/F1 and better context rich models for better results.
- **Runtime Notes**: GPU for Transformers; handle large transcripts by chunking.

