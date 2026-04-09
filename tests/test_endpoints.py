import sys
sys.path.insert(0, "/SPXvePFS/users/jytang/metabolm_posttrain")
from src.data.endpoints import (
    DISEASE_ENDPOINTS, get_disease_names, get_unique_chapters,
    get_chapter_to_diseases, get_disease_to_chapter_idx, get_all_icd10_codes
)

def test_endpoint_count():
    assert len(DISEASE_ENDPOINTS) == 16
    print("test_endpoint_count PASSED")

def test_disease_names():
    names = get_disease_names()
    assert len(names) == 16
    assert len(set(names)) == 16  # all unique
    assert "T2D" in names
    assert "hypertension" in names
    print("test_disease_names PASSED")

def test_unique_chapters():
    chapters = get_unique_chapters()
    # Should be: chapter_02, chapter_04, chapter_06, chapter_09, chapter_10, chapter_13
    assert len(chapters) == 6
    assert "chapter_02" in chapters  # cancers
    assert "chapter_09" in chapters  # circulatory
    print("test_unique_chapters PASSED")

def test_chapter_to_diseases():
    mapping = get_chapter_to_diseases()
    # chapter_09: hypertension, ischemic_heart, atrial_fib, heart_failure, stroke
    assert len(mapping["chapter_09"]) == 5
    print("test_chapter_to_diseases PASSED")

def test_disease_to_chapter_idx():
    mapping = get_disease_to_chapter_idx()
    assert len(mapping) == 16
    # All values should be in range [0, 5]
    assert all(0 <= v <= 5 for v in mapping.values())
    print("test_disease_to_chapter_idx PASSED")

def test_all_icd10_codes():
    codes = get_all_icd10_codes()
    assert "E11" in codes
    assert "I10" in codes
    assert "G30" in codes  # Alzheimer's (part of dementia)
    print("test_all_icd10_codes PASSED")

if __name__ == "__main__":
    test_endpoint_count()
    test_disease_names()
    test_unique_chapters()
    test_chapter_to_diseases()
    test_disease_to_chapter_idx()
    test_all_icd10_codes()
    print("All endpoint tests passed!")
