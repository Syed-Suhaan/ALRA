try:
    from src.ingest import load_and_process_pdfs
    print("Import src.ingest successful")
except Exception as e:
    print(f"Import src.ingest FAILED: {e}")

try:
    from src.engine import get_answer
    print("Import src.engine successful")
except Exception as e:
    print(f"Import src.engine FAILED: {e}")
