"""
설정 파일

BERTopic 분석에 필요한 모든 설정을 관리합니다.
"""

from typing import List, Dict, Any


class Config:
    """설정 클래스"""

    # OpenAI API 설정
    OPENAI_API_KEY: str = None  # 환경변수나 직접 설정 필요
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o-mini"

    # BERTopic 파라미터
    MIN_TOPIC_SIZE: int = 10
    MAX_TOPICS: int = 50
    MAX_NOISE_RATIO: float = 0.10

    # 임베딩 설정
    EMBEDDING_BATCH_SIZE: int = 100

    # Grid Search 설정
    SAMPLE_SIZE: int = 1000
    RANDOM_SEED: int = 42

    # UMAP 파라미터 그리드
    UMAP_GRID: List[Dict[str, Any]] = [
        {
            'n_neighbors': 30,
            'n_components': 10,
            'min_dist': 0.15,
            'metric': 'cosine',
            'random_state': 42
        }
    ]

    # HDBSCAN 파라미터 그리드
    HDBSCAN_GRID: Dict[str, List] = {
        'min_cluster_size': [20, 25, 30, 40],
        'min_samples': [6, 8, 10, 12],
        'epsilon': [0.05, 0.10, 0.20]
    }

    # 전처리 설정
    CUSTOM_STOPWORDS: List[str] = []

    # 출력 설정
    SHOW_PROGRESS: bool = True
    VERBOSE: bool = False

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """
        딕셔너리에서 설정 로드

        Args:
            config_dict: 설정 딕셔너리
        """
        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """
        설정을 딕셔너리로 변환

        Returns:
            설정 딕셔너리
        """
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith('_') and not callable(getattr(cls, key))
        }


# 간단한 사용을 위한 디폴트 설정 인스턴스
default_config = Config()
