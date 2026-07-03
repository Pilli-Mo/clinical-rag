import requests
import json
from pathlib import Path

# ClinicalTrials.gov API endpoint
BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


def fetch_trials(condition: str = "diabetes", max_results: int = 10) -> list:
    """Fetch clinical trial studies from ClinicalTrials.gov"""

    params = {
        "query.cond": condition,
        "pageSize": max_results,
        "format": "json",
        "fields": "NCTId,BriefTitle,BriefSummary,EligibilityCriteria,OverallStatus"
    }

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()   # errorr raised in case sthm goes wrong

    data = response.json()
    studies = data.get("studies", [])
    print(f"Fetched {len(studies)} studies for condition: {condition}")
    return studies


def extract_text(study: dict) -> dict:
    """Extract and clean relevant text fields from a study"""

    protocol = study.get("protocolSection", {})
    id_module = protocol.get("identificationModule", {})
    description_module = protocol.get("descriptionModule", {})
    eligibility_module = protocol.get("eligibilityModule", {})
    status_module = protocol.get("statusModule", {})

    return {
        "id": id_module.get("nctId", ""),
        "title": id_module.get("briefTitle", ""),
        "summary": description_module.get("briefSummary", ""),
        "eligibility": eligibility_module.get("eligibilityCriteria", ""),
        "status": status_module.get("overallStatus", "")
    }


def chunk_text(text: str, chunk_size: int = 200) -> list:
    """Split text into chunks of approximately chunk_size words"""

    words = text.split()
    chunks = []

    # Step through the words in chunk_size steps
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def process_trials(condition: str = "diabetes", max_results: int = 10) -> list:
    """Full pipeline: fetch → extract → chunk → return"""

    studies = fetch_trials(condition, max_results)
    all_chunks = []

    for study in studies:
        extracted = extract_text(study)

        # Combine summary and eligibility as the main text to chunk
        full_text = f"{extracted['summary']} {extracted['eligibility']}"
        chunks = chunk_text(full_text)

        # Each chunk keeps track of where it came from
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "trial_id": extracted["id"],
                "title": extracted["title"],
                "chunk_index": i,
                "text": chunk
            })

    print(f"Created {len(all_chunks)} chunks from {len(studies)} studies")
    return all_chunks


if __name__ == "__main__":
    chunks = process_trials(condition="diabetes", max_results=5)

    # Save to data folder
    output_path = Path("data/chunks.json")
    with open(output_path, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Saved chunks to {output_path}")
    print(f"\nSample chunk:")
    print(json.dumps(chunks[0], indent=2))
