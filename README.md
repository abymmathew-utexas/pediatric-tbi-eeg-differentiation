# Pediatric EEG AI for Differentiating Non-Accidental vs Accidental Traumatic Brain Injury

## Project Mentors
- Dr. Elias Rizk (Pediatric Neurosurgery)
- Dr. Chelsey Ortman (Pediatric Neurology)

## Collaboration Team
- Collaborators in Pediatric Neurology/Epilepsy and Pediatric Neurosurgery/Child Abuse Medicine
- Graduate students and residents with experience in clinical data research
- Potential collaboration with data science faculty
- Potential cross-institutional collaboration with faculty at Children's Hospital of Pittsburgh or other institutions depending on final project scope

## Project Description
This project explores how artificial intelligence can help clinicians differentiate between non-accidental and accidental traumatic brain injury using background EEG patterns.

Pediatric neurosurgeons and neurologists frequently encounter young children with new-onset seizures or altered mental status where the underlying cause may be ambiguous. Rapid and accurate differentiation is critical for timely treatment, legal implications, and child safety.

The team will develop and train machine learning models on de-identified EEG datasets to detect subtle background features that distinguish these conditions. Workstreams include:
- Data preprocessing (artifact removal, filtering, quality control)
- Feature extraction (time-domain, frequency-domain, time-frequency)
- Model training (deep learning and classical machine learning)
- Evaluation (sensitivity, specificity, AUROC, confusion matrices)

This project targets an area with profound clinical and ethical impact: protecting vulnerable children while advancing AI applications in pediatric neurosurgery and neurology.

## Potential Outcomes
- A trained AI model capable of differentiating child abuse, trauma, and seizure-related cases using EEG background data
- A reproducible EEG preprocessing and feature extraction pipeline
- A preliminary manuscript or abstract suitable for pediatric neurosurgery, neurology, or AI in Medicine conferences
- Cross-disciplinary collaboration across neurosurgery, neurology/epilepsy, and data science teams

## Skills Needed
- Strong programming background in Python
- Basic understanding of machine learning concepts
- Familiarity with EEG or biosignal data (preferred, not required)
- Ability to work with large, noisy datasets
- Curiosity and interest in medical AI applications

## Tools and Technologies
- Python for data analysis and machine learning
- TensorFlow or PyTorch for deep learning
- MNE-Python for EEG preprocessing and analysis
- Scikit-learn for traditional machine learning
- Signal processing methods (FFT, wavelet transforms)
- Git and GitHub for collaboration and version control

## Repository Structure
```
.github/               # CI workflows, issue templates, PR templates
data/                  # Data pointers only (no PHI or raw sensitive data)
docs/                  # Protocols, governance, and study documentation
notebooks/             # Exploratory analysis and prototyping
reports/               # Figures and project artifacts
src/                   # Core pipeline and model code
```

## Quick Start
1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Review `docs/project_charter.md` and `docs/data_governance.md` before any data work.

## Data Governance and Ethics
- Use only de-identified datasets and approved data-use workflows.
- Do not commit PHI, PII, or restricted datasets to this repository.
- Follow IRB, HIPAA, institutional, and sponsor requirements.
- Use this repository as a code and metadata workspace, not a clinical data storage system.

## Suggested Citation
If this repository contributes to publications, include mentors and collaborating teams according to institutional authorship guidance.

## License
This repository is provided for academic and research collaboration. Add your institution-approved license in `LICENSE`.
