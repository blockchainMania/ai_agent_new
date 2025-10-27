"""
BERTopic 모델링 모듈

토픽 모델링 및 최적 파라미터 탐색을 담당합니다.
- 임베딩 생성
- Grid Search를 통한 최적 파라미터 탐색
- BERTopic 모델 학습
"""

import time
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from itertools import product

from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from langchain_openai import OpenAIEmbeddings
from bertopic.representation import OpenAI
from openai import OpenAI as OpenAIClient


class BERTopicModel:
    """BERTopic 모델 래퍼 클래스"""

    def __init__(
        self,
        openai_api_key: str,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        min_topic_size: int = 10
    ):
        """
        Args:
            openai_api_key: OpenAI API 키
            embedding_model: 임베딩 모델명
            llm_model: 토픽 레이블 생성용 LLM 모델명
            min_topic_size: 최소 토픽 크기
        """
        self.api_key = openai_api_key
        self.embedding_model_name = embedding_model
        self.llm_model = llm_model
        self.min_topic_size = min_topic_size

        # 임베딩 모델 초기화
        self.embedding_model = OpenAIEmbeddings(
            model=embedding_model,
            api_key=openai_api_key
        )

        # LLM 클라이언트 초기화
        self.client = OpenAIClient(api_key=openai_api_key)

        # 표현 모델 초기화
        self.representation_model = OpenAI(
            client=self.client,
            model=llm_model,
            api_key=openai_api_key,
            verbose=False,
            chat=True
        )

        self.topic_model = None
        self.embeddings = None

    def create_embeddings(
        self,
        documents: List[str],
        batch_size: int = 100,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        문서 임베딩 생성

        Args:
            documents: 문서 리스트
            batch_size: 배치 크기
            show_progress: 진행률 표시 여부

        Returns:
            임베딩 배열 (n_docs, embedding_dim)
        """
        embeddings = []
        total_batches = (len(documents) + batch_size - 1) // batch_size

        iterator = range(0, len(documents), batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=total_batches,
                desc="Creating embeddings"
            )

        for i in iterator:
            batch = documents[i:i + batch_size]
            batch_embeddings = self.embedding_model.embed_documents(batch)
            embeddings.extend(batch_embeddings)

            if show_progress and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({
                    'Processed': f"{min(i + batch_size, len(documents))}/{len(documents)} docs"
                })

        self.embeddings = np.array(embeddings)
        return self.embeddings

    def _build_vectorizer(self, min_df: int = 2) -> CountVectorizer:
        """
        CountVectorizer 생성

        Args:
            min_df: 최소 문서 빈도

        Returns:
            CountVectorizer 객체
        """
        return CountVectorizer(
            max_df=0.95,
            min_df=min_df,
            ngram_range=(1, 1),
            token_pattern=r"(?u)[\w\-]+",
            strip_accents='unicode',
            lowercase=True
        )

    def _sanitize_docs(self, docs: List[str]) -> List[str]:
        """빈 문서를 placeholder로 대체"""
        return [d.strip() if d.strip() else "placeholdertoken" for d in docs]

    def grid_search(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        sample_size: int = 1000,
        umap_grid: Optional[List[dict]] = None,
        hdbscan_grid: Optional[Dict[str, List]] = None,
        max_noise: float = 0.10,
        max_topics: int = 50,
        seed: int = 42,
        show_progress: bool = True
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Grid Search로 최적 파라미터 탐색

        Args:
            documents: 문서 리스트
            embeddings: 임베딩 배열
            sample_size: 샘플링 크기
            umap_grid: UMAP 파라미터 그리드
            hdbscan_grid: HDBSCAN 파라미터 그리드
            max_noise: 최대 노이즈 비율
            max_topics: 최대 토픽 개수
            seed: 랜덤 시드
            show_progress: 진행률 표시 여부

        Returns:
            (최적 파라미터, 전체 결과 DataFrame)
        """
        from .metrics import TopicMetrics

        # 샘플링
        rng = np.random.default_rng(seed)
        sample_size = min(sample_size, len(documents))
        idx = rng.choice(len(documents), size=sample_size, replace=False)
        docs_sample = [documents[i] for i in idx]
        emb_sample = embeddings[idx]

        # 기본 그리드 설정
        if umap_grid is None:
            umap_grid = [
                dict(n_neighbors=30, n_components=10, min_dist=0.15,
                     metric="cosine", random_state=seed)
            ]

        if hdbscan_grid is None:
            hdbscan_grid = {
                'min_cluster_size': [20, 25, 30, 40],
                'min_samples': [6, 8, 10, 12],
                'epsilon': [0.05, 0.10, 0.20]
            }

        # Vectorizer 설정
        min_df = 1 if len(docs_sample) < 100 else 2
        vectorizer = self._build_vectorizer(min_df=min_df)
        docs_sample = self._sanitize_docs(docs_sample)

        # 조합 생성
        combos = list(product(
            range(len(umap_grid)),
            hdbscan_grid['min_cluster_size'],
            hdbscan_grid['min_samples'],
            hdbscan_grid['epsilon']
        ))

        if show_progress:
            combos = tqdm(combos, desc="Grid Search", ncols=100)

        results = []
        umap_cache = {}

        for ui, mcs, ms, eps in combos:
            umap_params = umap_grid[ui].copy()

            # UMAP n_neighbors 제약
            n_samples = emb_sample.shape[0]
            if n_samples > 2:
                umap_params["n_neighbors"] = min(
                    umap_params["n_neighbors"],
                    max(2, n_samples - 1)
                )

            # UMAP 캐시
            if ui not in umap_cache:
                try:
                    umap_model = UMAP(**umap_params)
                    umap_cache[ui] = umap_model.fit_transform(emb_sample)
                except Exception as e:
                    results.append({
                        'umap_idx': ui,
                        'min_cluster_size': mcs,
                        'min_samples': ms,
                        'epsilon': eps,
                        'error': str(e),
                        'feasible': False
                    })
                    continue

            reduced_embeddings = umap_cache[ui]

            try:
                # HDBSCAN + BERTopic
                hdbscan_model = HDBSCAN(
                    min_cluster_size=mcs,
                    min_samples=ms,
                    cluster_selection_epsilon=eps,
                    metric="euclidean",
                    prediction_data=False
                )

                topic_model = BERTopic(
                    embedding_model=None,
                    umap_model=None,
                    hdbscan_model=hdbscan_model,
                    min_topic_size=self.min_topic_size,
                    vectorizer_model=vectorizer,
                    verbose=False
                )

                topics, _ = topic_model.fit_transform(
                    docs_sample,
                    embeddings=reduced_embeddings
                )

                # 평가 지표 계산
                info = topic_model.get_topic_info()
                metrics = TopicMetrics.calculate_metrics(info)

                # 제약 조건 확인
                feasible = (
                    metrics['outlier_ratio'] < max_noise and
                    metrics['n_topics_clean'] <= max_topics
                )

                results.append({
                    'umap_idx': ui,
                    'umap_params': umap_params,
                    'min_cluster_size': mcs,
                    'min_samples': ms,
                    'epsilon': eps,
                    **metrics,
                    'score': TopicMetrics.calculate_score(metrics),
                    'feasible': feasible,
                    'model': topic_model if feasible else None
                })

            except Exception as e:
                results.append({
                    'umap_idx': ui,
                    'min_cluster_size': mcs,
                    'min_samples': ms,
                    'epsilon': eps,
                    'error': str(e),
                    'feasible': False
                })

        # 결과 정리
        df = pd.DataFrame(results)

        # 최적 파라미터 선택 (feasible 중 score 최대)
        feasible_df = df[df['feasible'] == True]

        if not feasible_df.empty:
            best_idx = feasible_df['score'].idxmax()
            best_params = df.loc[best_idx].to_dict()
        else:
            # feasible 없으면 score 최대값
            best_idx = df['score'].idxmax() if 'score' in df.columns else 0
            best_params = df.loc[best_idx].to_dict()

        return best_params, df

    def fit(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        umap_params: Dict[str, Any],
        hdbscan_params: Dict[str, Any],
        verbose: bool = False
    ) -> BERTopic:
        """
        BERTopic 모델 학습

        Args:
            documents: 문서 리스트
            embeddings: 임베딩 배열
            umap_params: UMAP 파라미터
            hdbscan_params: HDBSCAN 파라미터
            verbose: 로그 출력 여부

        Returns:
            학습된 BERTopic 모델
        """
        # UMAP n_neighbors 제약
        n_samples = embeddings.shape[0]
        if n_samples > 2:
            umap_params["n_neighbors"] = min(
                umap_params["n_neighbors"],
                max(2, n_samples - 1)
            )

        # Vectorizer 설정
        min_df = 1 if len(documents) < 100 else 2
        vectorizer = self._build_vectorizer(min_df=min_df)

        # 문서 전처리
        documents = self._sanitize_docs(documents)

        if verbose:
            print(f"[Fit] UMAP params: {umap_params}")
            print(f"[Fit] HDBSCAN params: {hdbscan_params}")

        # UMAP
        umap_model = UMAP(**umap_params)
        reduced_embeddings = umap_model.fit_transform(embeddings)

        # HDBSCAN
        hdbscan_model = HDBSCAN(**hdbscan_params)

        # BERTopic
        self.topic_model = BERTopic(
            embedding_model=None,
            umap_model=None,
            hdbscan_model=hdbscan_model,
            min_topic_size=self.min_topic_size,
            vectorizer_model=vectorizer,
            verbose=verbose
        )

        topics, probs = self.topic_model.fit_transform(
            documents,
            embeddings=reduced_embeddings
        )

        if verbose:
            info = self.topic_model.get_topic_info()
            from .metrics import TopicMetrics
            metrics = TopicMetrics.calculate_metrics(info)
            print(f"[Fit] Topics: {metrics['n_topics_clean']}, "
                  f"Noise: {metrics['outlier_ratio']:.3f}")

        return self.topic_model

    def save_model(self, path: str):
        """모델 저장"""
        if self.topic_model is None:
            raise ValueError("No model to save. Please fit the model first.")

        self.topic_model.save(path, serialization="pickle")

    @staticmethod
    def load_model(path: str) -> BERTopic:
        """모델 로드"""
        return BERTopic.load(path)
