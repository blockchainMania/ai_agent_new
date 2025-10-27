"""
토픽 모델 평가 지표 모듈

토픽 모델의 품질을 평가하는 다양한 지표를 제공합니다.
- 노이즈 비율 (outlier_ratio)
- 지니 계수 (gini_clean)
- 최대 토픽 비율 (top1_clean)
- 정규화 엔트로피 (h_norm)
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any


class TopicMetrics:
    """토픽 모델 평가 지표 클래스"""

    @staticmethod
    def calculate_gini(values: np.ndarray) -> float:
        """
        지니 계수 계산 (토픽 분포 불균등도)

        Args:
            values: 토픽별 문서 개수 배열

        Returns:
            지니 계수 (0: 균등, 1: 극단적 불균등)
        """
        x = np.asarray(values, dtype=float).ravel()
        n = x.size

        if n == 0:
            return 0.0

        s = x.sum()
        if s <= 0:
            return 0.0

        x.sort()
        cum = np.cumsum(x)

        return float((n + 1 - 2 * np.sum(cum) / s) / n)

    @staticmethod
    def calculate_entropy(counts: np.ndarray, normalize: bool = True) -> float:
        """
        엔트로피 계산

        Args:
            counts: 토픽별 문서 개수 배열
            normalize: 정규화 여부 (True: 0~1 범위)

        Returns:
            엔트로피 값
        """
        s = counts.sum()
        if s <= 0:
            return 0.0

        p = counts / s
        p = p[p > 0]  # 0 제거

        entropy = float(-np.sum(p * np.log(p)))

        if normalize and len(p) > 1:
            max_entropy = math.log(len(p))
            entropy = entropy / max_entropy

        return entropy

    @staticmethod
    def calculate_metrics(topic_info: pd.DataFrame) -> Dict[str, Any]:
        """
        토픽 모델의 모든 평가 지표 계산

        Args:
            topic_info: BERTopic의 get_topic_info() 결과

        Returns:
            평가 지표 딕셔너리
            - outlier_ratio: 노이즈 비율
            - gini_clean: 지니 계수 (노이즈 제외)
            - top1_clean: 최대 토픽 비율 (노이즈 제외)
            - n_topics_clean: 유효 토픽 개수 (노이즈 제외)
            - h_norm: 정규화 엔트로피
        """
        if topic_info is None or topic_info.empty:
            return {
                'outlier_ratio': np.nan,
                'gini_clean': np.nan,
                'top1_clean': np.nan,
                'n_topics_clean': 0,
                'h_norm': np.nan
            }

        # 전체 문서 수
        total_docs = int(topic_info["Count"].sum())

        # 노이즈 비율 계산
        outlier_count = 0
        if -1 in topic_info.Topic.values:
            outlier_count = int(
                topic_info.loc[topic_info.Topic == -1, "Count"].sum()
            )

        outlier_ratio = (
            outlier_count / total_docs if total_docs > 0 else np.nan
        )

        # 노이즈 제외한 유효 토픽
        clean_topics = topic_info[topic_info.Topic != -1]

        if clean_topics.empty:
            return {
                'outlier_ratio': outlier_ratio,
                'gini_clean': np.nan,
                'top1_clean': np.nan,
                'n_topics_clean': 0,
                'h_norm': np.nan
            }

        counts = clean_topics["Count"].values.astype(float)
        counts_sum = counts.sum()

        if counts_sum <= 0:
            return {
                'outlier_ratio': outlier_ratio,
                'gini_clean': 0.0,
                'top1_clean': np.nan,
                'n_topics_clean': int(len(counts)),
                'h_norm': np.nan
            }

        # 확률 분포
        p = counts / counts_sum

        # 지니 계수
        gini = TopicMetrics.calculate_gini(counts)

        # 최대 토픽 비율
        top1 = float(p.max())

        # 정규화 엔트로피
        h_norm = TopicMetrics.calculate_entropy(counts, normalize=True)

        return {
            'outlier_ratio': float(outlier_ratio),
            'gini_clean': float(gini),
            'top1_clean': float(top1),
            'n_topics_clean': int(len(counts)),
            'h_norm': float(h_norm)
        }

    @staticmethod
    def calculate_score(metrics: Dict[str, Any]) -> float:
        """
        종합 점수 계산 (높을수록 좋음)

        Args:
            metrics: calculate_metrics()의 결과

        Returns:
            종합 점수 (0~1)
        """
        # 노이즈 비율: 낮을수록 좋음
        outlier_term = 1.0 - float(metrics.get('outlier_ratio', 0.5))
        if not np.isfinite(outlier_term):
            outlier_term = 0.5

        # 정규화 엔트로피: 높을수록 좋음 (균등 분포)
        h_norm = float(metrics.get('h_norm', 0.5))
        if not np.isfinite(h_norm):
            h_norm = 0.5

        # 최대 토픽 비율: 너무 크면 감점
        top1 = float(metrics.get('top1_clean', 0.5))
        if not np.isfinite(top1):
            top1 = 0.5
        # 40% 이상이면 감점
        top1_term = 1.0 - max(0.0, (top1 - 0.40) / 0.60)
        top1_term = max(0.0, min(1.0, top1_term))

        # 지니 계수: 낮을수록 좋음
        gini = float(metrics.get('gini_clean', 0.5))
        if not np.isfinite(gini):
            gini = 0.5
        gini_term = 1.0 - max(0.0, min(1.0, gini))

        # 가중 평균
        score = (
            0.35 * outlier_term +
            0.25 * h_norm +
            0.25 * top1_term +
            0.15 * gini_term
        )

        return float(score)

    @staticmethod
    def print_metrics(metrics: Dict[str, Any]):
        """
        평가 지표 출력

        Args:
            metrics: calculate_metrics()의 결과
        """
        print("\n=== Topic Model Metrics ===")
        print(f"유효 토픽 개수: {metrics['n_topics_clean']}")
        print(f"노이즈 비율: {metrics['outlier_ratio']:.3f}")
        print(f"정규화 엔트로피: {metrics['h_norm']:.3f}")
        print(f"최대 토픽 비율: {metrics['top1_clean']:.3f}")
        print(f"지니 계수: {metrics['gini_clean']:.3f}")
        print(f"종합 점수: {TopicMetrics.calculate_score(metrics):.3f}")
        print("=" * 30)
