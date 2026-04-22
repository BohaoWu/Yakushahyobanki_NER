#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation and Visualization Module

Performs evaluation, comparative analysis, and visualization based on multiple HistoryDocumentResult objects.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json


class ResultEvaluator:
    """Evaluator based on multiple HistoryDocumentResult objects"""

    def __init__(self, results: List['HistoryDocumentResult']):
        """
        Initialize the evaluator

        Args:
            results: List of HistoryDocumentResult objects
        """
        from src.core.train import HistoryDocumentResult  # Lazy import to avoid circular dependency

        self.all_results = results
        self.successful_results = [r for r in results if r.status == "success"]

        if not self.successful_results:
            raise ValueError("No successful experiment results")

    @classmethod
    def from_files(cls, result_files: List[str]) -> 'ResultEvaluator':
        """
        Create an evaluator from multiple result.json files

        Args:
            result_files: List of result.json file paths

        Returns:
            ResultEvaluator instance
        """
        from src.core.train import HistoryDocumentResult

        results = []
        for file_path in result_files:
            try:
                result = HistoryDocumentResult.load(file_path)
                results.append(result)
            except Exception as e:
                print(f"Warning: Unable to load {file_path}: {str(e)}")

        return cls(results)

    @classmethod
    def from_directory(cls, directory: str, pattern: str = "*/result.json") -> 'ResultEvaluator':
        """
        Load all result files from a directory

        Args:
            directory: Directory path
            pattern: File matching pattern

        Returns:
            ResultEvaluator instance
        """
        dir_path = Path(directory)
        result_files = list(dir_path.glob(pattern))

        if not result_files:
            raise FileNotFoundError(f"No files matching {pattern} found in {directory}")

        print(f"Found {len(result_files)} result files")
        return cls.from_files([str(f) for f in result_files])

    def get_best_model(self, metric: str = "f1") -> 'HistoryDocumentResult':
        """
        Get the best model

        Args:
            metric: Evaluation metric

        Returns:
            HistoryDocumentResult of the best model
        """
        return max(self.successful_results, key=lambda r: r.metrics.get(metric, 0))

    def get_worst_model(self, metric: str = "f1") -> 'HistoryDocumentResult':
        """Get the worst model"""
        return min(self.successful_results, key=lambda r: r.metrics.get(metric, 0))

    def get_ranking(self, metric: str = "f1") -> List[Tuple[str, float]]:
        """
        Get model ranking

        Args:
            metric: Sorting metric

        Returns:
            [(model_key, score), ...] list
        """
        rankings = [
            (r.model_key, r.metrics.get(metric, 0))
            for r in self.successful_results
        ]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_summary_table(self) -> Dict:
        """
        Get summary table data

        Returns:
            Dictionary containing all model metrics
        """
        summary = []
        for r in sorted(self.successful_results, key=lambda x: x.metrics["f1"], reverse=True):
            # Check model characteristics
            model_name_lower = (r.model_name or "").lower()
            model_key_lower = (r.model_key or "").lower()

            use_crf = r.hyperparameters.get("use_crf", False) or "_crf" in model_key_lower
            is_large = "large" in model_name_lower or "large" in model_key_lower
            is_bilstm = "bilstm" in model_key_lower or "bilstm" in model_name_lower
            is_llama = any(x in model_key_lower or x in model_name_lower for x in ["llama", "rinna", "gpt-neox", "elyza"])
            is_remote_llm = any(x in model_key_lower for x in ["chatgpt", "claude", "remote"])

            # Determine model type
            if is_bilstm:
                model_type = "bilstm"
            elif is_remote_llm:
                model_type = "remote_llm"
            elif is_llama:
                model_type = "llama"
            else:
                model_type = "bert"

            summary.append({
                "model_key": r.model_key,
                "display_name": self._format_model_display_name(r),
                "model_name": r.model_name,
                "model_type": model_type,
                "use_crf": use_crf,
                "is_large": is_large,
                "f1": r.metrics["f1"],
                "precision": r.metrics["precision"],
                "recall": r.metrics["recall"],
                "training_time": r.training_time,
                "num_train_samples": r.num_train_samples,
                "num_test_samples": r.num_test_samples,
            })
        return summary

    def _format_model_display_name(self, result) -> str:
        """
        Format model display name, ensuring CRF, large, and model type are clearly shown

        Args:
            result: HistoryDocumentResult object

        Returns:
            Formatted display name
        """
        model_key = result.model_key or ""
        model_name = (result.model_name or "").lower()

        # Base name (prioritize model_key)
        display_name = model_key

        # Check if CRF is used (determined from hyperparameters or model_key)
        use_crf = result.hyperparameters.get("use_crf", False)
        has_crf_in_key = "_crf" in model_key.lower() or "-crf" in model_key.lower()

        # Check if it is a large model
        is_large = "large" in model_name or "large" in model_key.lower()
        has_large_in_key = "large" in model_key.lower()

        # Check model type
        is_bilstm = "bilstm" in model_key.lower() or "bilstm" in model_name
        is_llama = any(x in model_key.lower() or x in model_name for x in ["llama", "rinna", "gpt-neox", "elyza"])
        is_remote_llm = any(x in model_key.lower() for x in ["chatgpt", "claude", "remote"])

        # Add suffix (if not already in model_key)
        if is_large and not has_large_in_key:
            display_name = f"{display_name}-large"
        if use_crf and not has_crf_in_key:
            display_name = f"{display_name}-crf"

        # Add model type flags (in square brackets)
        flags = []
        if is_bilstm and "bilstm" not in model_key.lower():
            flags.append("BiLSTM")
        elif is_llama and not any(x in model_key.lower() for x in ["llama", "rinna"]):
            flags.append("LLM")
        elif is_remote_llm and "remote" not in model_key.lower():
            flags.append("API")

        if flags:
            display_name = f"{display_name} [{'/'.join(flags)}]"

        return display_name

    def print_summary(self):
        """Print summary table"""
        print("\n" + "=" * 110)
        print(f"Evaluation Summary ({len(self.successful_results)} models)")
        print("=" * 110)
        print(f"{'Rank':<5} {'Model':<30} {'F1':>10} {'Precision':>10} {'Recall':>10} "
              f"{'Train Time(s)':>12}")
        print("-" * 110)

        for i, r in enumerate(sorted(self.successful_results,
                                     key=lambda x: x.metrics["f1"], reverse=True), 1):
            icon = "🏆" if i == 1 else "  "
            display_name = self._format_model_display_name(r)
            print(f"{icon}{i:<4} {display_name:<30} {r.metrics['f1']:>10.4f} "
                  f"{r.metrics['precision']:>10.4f} {r.metrics['recall']:>10.4f} "
                  f"{r.training_time:>12.1f}")
        print("=" * 110)

        # Best model
        best = self.get_best_model()
        best_display = self._format_model_display_name(best)
        print(f"\n🏆 Best Model: {best_display} (F1: {best.metrics['f1']:.4f})")

    def compare_entity_types(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compare performance of different models across entity types

        Returns:
            {entity_type: [(model_key, f1_score), ...], ...}
        """
        # Collect all entity types
        all_entity_types = set()
        for r in self.successful_results:
            all_entity_types.update(r.metrics_per_type.keys())

        # For each entity type, collect scores from all models
        entity_comparison = {}
        for entity_type in all_entity_types:
            scores = []
            for r in self.successful_results:
                if entity_type in r.metrics_per_type:
                    score = r.metrics_per_type[entity_type].get("f1-score", 0)
                    scores.append((r.model_key, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            entity_comparison[entity_type] = scores

        return entity_comparison

    def find_best_model_per_entity(self) -> Dict[str, str]:
        """
        Find the best model for each entity type

        Returns:
            {entity_type: model_key, ...}
        """
        entity_comparison = self.compare_entity_types()
        return {
            entity_type: scores[0][0]
            for entity_type, scores in entity_comparison.items()
            if scores
        }

    def plot_all_metrics_comparison(self, save_dir: str = None, show: bool = False):
        """
        Plot comparison charts for all metrics

        Args:
            save_dir: Save directory
            show: Whether to display
        """
        from src.core.train import HistoryDocumentResult

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        else:
            save_path = Path(".")

        print("\nGenerating comparison charts...")

        # F1 comparison
        HistoryDocumentResult.compare_multiple(
            self.successful_results,
            metric="f1",
            save_path=str(save_path / "f1_comparison.png"),
            show=show
        )

        # Precision comparison
        HistoryDocumentResult.compare_multiple(
            self.successful_results,
            metric="precision",
            save_path=str(save_path / "precision_comparison.png"),
            show=show
        )

        # Recall comparison
        HistoryDocumentResult.compare_multiple(
            self.successful_results,
            metric="recall",
            save_path=str(save_path / "recall_comparison.png"),
            show=show
        )

        # Training time comparison
        HistoryDocumentResult.plot_training_time_comparison(
            self.successful_results,
            save_path=str(save_path / "training_time_comparison.png"),
            show=show
        )

        print(f"✓ Charts saved to: {save_path}")

    def plot_entity_type_comparison(self, entity_type: str, save_path: str = None, show: bool = False):
        """
        Compare performance of different models on a specific entity type

        Args:
            entity_type: Entity type
            save_path: Save path
            show: Whether to display
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
            plt.rcParams['axes.unicode_minus'] = False
        except ImportError:
            print("Error: matplotlib needs to be installed")
            return

        # Collect data
        model_names = []
        f1_scores = []
        precision_scores = []
        recall_scores = []

        for r in self.successful_results:
            if entity_type in r.metrics_per_type:
                metrics = r.metrics_per_type[entity_type]
                model_names.append(self._format_model_display_name(r))
                f1_scores.append(metrics.get("f1-score", 0))
                precision_scores.append(metrics.get("precision", 0))
                recall_scores.append(metrics.get("recall", 0))

        if not model_names:
            print(f"Warning: No model contains entity type {entity_type}")
            return

        # Plot grouped bar chart
        import numpy as np

        x = np.arange(len(model_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precision_scores, width, label='Precision', color='#3498db')
        ax.bar(x, recall_scores, width, label='Recall', color='#2ecc71')
        ax.bar(x + width, f1_scores, width, label='F1', color='#e74c3c')

        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Performance Comparison on Entity Type: {entity_type}',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Charts saved to: {save_path}")

        if show:
            plt.show()
        plt.close()

    def plot_precision_recall_scatter(self, save_path: str = None, show: bool = False):
        """
        Plot Precision-Recall scatter plot

        Args:
            save_path: Save path
            show: Whether to display
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            print("Error: matplotlib needs to be installed")
            return

        import numpy as np

        precisions = [r.metrics["precision"] for r in self.successful_results]
        recalls = [r.metrics["recall"] for r in self.successful_results]
        model_names = [self._format_model_display_name(r) for r in self.successful_results]

        # Dynamically calculate axis range to make differences more visible
        min_precision, max_precision = min(precisions), max(precisions)
        min_recall, max_recall = min(recalls), max(recalls)

        # Calculate data range, add margins
        precision_range = max_precision - min_precision
        recall_range = max_recall - min_recall

        # Set margins (30% of data range or at least 0.05)
        precision_margin = max(precision_range * 0.3, 0.05)
        recall_margin = max(recall_range * 0.3, 0.05)

        x_min = max(0, min_recall - recall_margin)
        x_max = min(1, max_recall + recall_margin)
        y_min = max(0, min_precision - precision_margin)
        y_max = min(1, max_precision + precision_margin)

        # Use different colors to distinguish models
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#9b59b6']

        plt.figure(figsize=(10, 8))

        # Plot different colored points for each model
        for i, (recall, precision, name) in enumerate(zip(recalls, precisions, model_names)):
            plt.scatter(recall, precision, s=300, alpha=0.7, c=colors[i % len(colors)],
                       label=name, edgecolors='white', linewidths=2)

        # Add model labels
        for i, name in enumerate(model_names):
            plt.annotate(name, (recalls[i], precisions[i]),
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=10, fontweight='bold', alpha=0.9)

        # Add F1 iso-lines (only show within visible range)
        # Calculate F1 values within visible range
        f1_values = [2 * p * r / (p + r) if (p + r) > 0 else 0
                     for p, r in zip(precisions, recalls)]
        min_f1 = min(f1_values)
        max_f1 = max(f1_values)

        # Select appropriate F1 iso-lines
        all_f1_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        f1_levels = [f for f in all_f1_levels if min_f1 - 0.1 <= f <= max_f1 + 0.1]
        if not f1_levels:
            f1_levels = [round(min_f1, 1), round((min_f1 + max_f1) / 2, 2), round(max_f1, 1)]

        for f1 in f1_levels:
            recall_line = np.linspace(max(0.01, x_min), x_max, 100)
            precision_line = (f1 * recall_line) / (2 * recall_line - f1)
            # Only plot the portion within the visible range
            mask = (precision_line >= y_min) & (precision_line <= y_max)
            if np.any(mask):
                plt.plot(recall_line[mask], precision_line[mask], '--', alpha=0.4, color='gray', linewidth=1)
                # Add label at the end of the line
                valid_indices = np.where(mask)[0]
                if len(valid_indices) > 0:
                    idx = valid_indices[-1]
                    if precision_line[idx] <= y_max and precision_line[idx] >= y_min:
                        plt.text(recall_line[idx] + 0.01, precision_line[idx], f'F1={f1}',
                                fontsize=9, alpha=0.6)

        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Scatter Plot\n(Scale: Recall {x_min:.2f}-{x_max:.2f}, Precision {y_min:.2f}-{y_max:.2f})',
                 fontsize=14, fontweight='bold')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend(loc='best', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Charts saved to: {save_path}")

        if show:
            plt.show()
        plt.close()

    def plot_radar_chart(self, models: List[str] = None, save_path: str = None, show: bool = False):
        """
        Plot radar chart comparing multiple models

        Args:
            models: List of models to compare (defaults to Top 5)
            save_path: Save path
            show: Whether to display
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            import numpy as np
        except ImportError:
            print("Error: matplotlib and numpy need to be installed")
            return

        # Select models to compare
        if models is None:
            # Display all models by default (sorted by F1)
            sorted_results = sorted(self.successful_results,
                                   key=lambda r: r.metrics["f1"], reverse=True)
            selected_results = sorted_results
        else:
            selected_results = [r for r in self.successful_results if r.model_key in models]

        if not selected_results:
            print("Warning: No matching models found")
            return

        # Define metrics
        metrics = ['Precision', 'Recall', 'F1']
        num_vars = len(metrics)

        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

        # Extend color list to support more models
        colors = [
            '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
            '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b',
            '#27ae60', '#2980b9', '#8e44ad', '#f1c40f', '#d35400',
            '#7f8c8d', '#2c3e50', '#1e90ff', '#ff6347', '#32cd32'
        ]

        # First collect all values to calculate dynamic range
        all_values = []
        for result in selected_results:
            all_values.extend([
                result.metrics["precision"],
                result.metrics["recall"],
                result.metrics["f1"],
            ])

        # Dynamically calculate Y-axis range to make differences more visible
        min_val = min(all_values)
        max_val = max(all_values)
        # Set Y-axis lower bound slightly below the minimum (with 10% margin), but not below 0
        y_min = max(0, min_val - (max_val - min_val) * 0.3 - 0.05)
        # Ensure there is at least a certain range difference
        y_min = min(y_min, min_val - 0.05)
        y_min = max(0, y_min)  # Cannot be negative
        y_max = min(1, max_val + 0.02)  # Leave a small margin at the upper limit

        for i, result in enumerate(selected_results):
            values = [
                result.metrics["precision"],
                result.metrics["recall"],
                result.metrics["f1"],
            ]
            values += values[:1]  # Close the polygon

            display_name = self._format_model_display_name(result)
            ax.plot(angles, values, 'o-', linewidth=2,
                   label=display_name, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylim(y_min, y_max)

        # Add value labels, show actual range
        ax.set_title(f'Model Performance Comparison (Radar Chart)\n(Scale: {y_min:.2f} - {y_max:.2f})',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Charts saved to: {save_path}")

        if show:
            plt.show()
        plt.close()

    def plot_entity_radar_chart(self, models: List[str] = None, save_path: str = None,
                                 show: bool = False, max_models: int = 8):
        """
        Plot radar chart comparing F1 scores across entity categories (entity types)

        Args:
            models: List of models to compare (defaults to top max_models sorted by F1)
            save_path: Save path
            show: Whether to display
            max_models: Maximum number of models to display
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            import numpy as np
        except ImportError:
            print("Error: matplotlib and numpy need to be installed")
            return

        # Set Japanese font
        plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans', 'sans-serif']

        # Select models to compare
        if models is None:
            sorted_results = sorted(self.successful_results,
                                   key=lambda r: r.metrics.get("f1", 0), reverse=True)
            selected_results = sorted_results[:max_models]
        else:
            selected_results = [r for r in self.successful_results if r.model_key in models]

        if not selected_results:
            print("Warning: No matching models found")
            return

        # Collect all entity types
        entity_types = set()
        for result in selected_results:
            if hasattr(result, 'metrics_per_type') and result.metrics_per_type:
                entity_types.update(result.metrics_per_type.keys())

        if not entity_types:
            print("Warning: No entity type data found")
            return

        entity_types = sorted(list(entity_types))
        num_entities = len(entity_types)

        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, num_entities, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        # Create chart
        fig, ax = plt.subplots(figsize=(14, 11), subplot_kw=dict(polar=True))

        # Color mapping
        colors = [
            '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
            '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b',
            '#27ae60', '#2980b9', '#8e44ad', '#f1c40f', '#d35400'
        ]

        for idx, result in enumerate(selected_results):
            per_type = getattr(result, 'metrics_per_type', {}) or {}

            # Get F1 score for each entity type
            values = []
            for entity in entity_types:
                f1 = per_type.get(entity, {}).get("f1-score", 0)
                values.append(f1)

            values += values[:1]  # Close the polygon

            display_name = self._format_model_display_name(result)

            # Plot
            ax.plot(angles, values, 'o-', linewidth=2,
                   label=f'{display_name} (F1={result.metrics.get("f1", 0):.3f})',
                   color=colors[idx % len(colors)], markersize=4)
            ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(entity_types, fontsize=10)

        # Dynamically calculate Y-axis range to make data differences more visible
        all_values = []
        for result in selected_results:
            per_type = getattr(result, 'metrics_per_type', {}) or {}
            for entity in entity_types:
                f1 = per_type.get(entity, {}).get("f1-score", 0)
                all_values.append(f1)

        if all_values:
            max_val = max(all_values)
            # Set Y-axis upper limit to max value + 10% margin, rounded up to 0.1
            y_max = min(1.0, np.ceil((max_val + 0.05) * 10) / 10)
            y_max = max(y_max, 0.3)  # At least 0.3
        else:
            y_max = 1.0

        ax.set_ylim(0, y_max)
        # Dynamically set tick marks
        num_ticks = 5
        tick_step = y_max / num_ticks
        yticks = [tick_step * i for i in range(1, num_ticks + 1)]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{t:.1f}' for t in yticks], fontsize=9)

        # Title and legend
        ax.set_title(f'Entity Category F1 Score Comparison by Model\n(Scale: 0-{y_max:.1f})',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Entity type radar chart saved to: {save_path}")

        if show:
            plt.show()
        plt.close()

    def plot_entity_radar_chart_crf_comparison(self, save_path: str = None, show: bool = False):
        """
        Plot radar chart comparing entity type F1 scores for CRF vs non-CRF models (dual chart)

        Args:
            save_path: Save path
            show: Whether to display
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            import numpy as np
        except ImportError:
            print("Error: matplotlib and numpy need to be installed")
            return

        plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans', 'sans-serif']

        # Separate CRF and non-CRF models
        crf_results = [r for r in self.successful_results if getattr(r, 'use_crf', False) or '_crf' in r.model_key]
        non_crf_results = [r for r in self.successful_results if not (getattr(r, 'use_crf', False) or '_crf' in r.model_key)]

        # Collect all entity types
        entity_types = set()
        for result in self.successful_results:
            if hasattr(result, 'metrics_per_type') and result.metrics_per_type:
                entity_types.update(result.metrics_per_type.keys())

        if not entity_types:
            print("Warning: No entity type data found")
            return

        entity_types = sorted(list(entity_types))
        num_entities = len(entity_types)

        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, num_entities, endpoint=False).tolist()
        angles += angles[:1]

        fig, axes = plt.subplots(1, 2, figsize=(20, 9), subplot_kw=dict(polar=True))

        colors = [
            '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
            '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b'
        ]

        # Calculate global maximum for unified Y-axis range
        all_values = []
        for result in self.successful_results:
            per_type = getattr(result, 'metrics_per_type', {}) or {}
            for entity in entity_types:
                f1 = per_type.get(entity, {}).get("f1-score", 0)
                all_values.append(f1)

        if all_values:
            max_val = max(all_values)
            y_max = min(1.0, np.ceil((max_val + 0.05) * 10) / 10)
            y_max = max(y_max, 0.3)
        else:
            y_max = 1.0

        for ax, (title, results) in zip(axes, [("Non-CRF Models", non_crf_results), ("CRF Models", crf_results)]):
            # Sort by F1, select top 6
            sorted_results = sorted(results, key=lambda r: r.metrics.get("f1", 0), reverse=True)[:6]

            for idx, result in enumerate(sorted_results):
                per_type = getattr(result, 'metrics_per_type', {}) or {}
                values = [per_type.get(entity, {}).get("f1-score", 0) for entity in entity_types]
                values += values[:1]

                display_name = self._format_model_display_name(result)
                ax.plot(angles, values, 'o-', linewidth=2,
                       label=f'{display_name} ({result.metrics.get("f1", 0):.3f})',
                       color=colors[idx % len(colors)], markersize=4)
                ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(entity_types, fontsize=9)
            ax.set_ylim(0, y_max)
            # Dynamically set tick marks
            num_ticks = 4
            tick_step = y_max / num_ticks
            yticks = [tick_step * i for i in range(1, num_ticks + 1)]
            ax.set_yticks(yticks)
            ax.set_yticklabels([f'{t:.2f}' for t in yticks], fontsize=8)
            ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Entity Category F1 Score Comparison (CRF vs Non-CRF, Scale: 0-{y_max:.1f})',
                    fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ CRF comparison radar chart saved to: {save_path}")

        if show:
            plt.show()
        plt.close()

    def export_to_csv(self, output_file: str):
        """
        Export to CSV file

        Args:
            output_file: Output file path
        """
        import pandas as pd

        summary = self.get_summary_table()
        df = pd.DataFrame(summary)

        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"✓ Exported to: {output_file}")

    def export_to_json(self, output_file: str):
        """
        Export to JSON file

        Args:
            output_file: Output file path
        """
        summary = self.get_summary_table()

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"✓ Exported to: {output_file}")

    def create_full_report(self, output_dir: str):
        """
        Create full evaluation report

        Args:
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Creating full evaluation report")
        print(f"{'='*80}")

        # 1. Print summary
        self.print_summary()

        # 2. Export CSV and JSON
        print("\nExporting data...")
        self.export_to_csv(str(output_path / "summary.csv"))
        self.export_to_json(str(output_path / "summary.json"))

        # 3. Generate all comparison charts
        self.plot_all_metrics_comparison(save_dir=str(output_path), show=False)

        # 4. Precision-Recall scatter plot
        self.plot_precision_recall_scatter(
            save_path=str(output_path / "precision_recall_scatter.png"),
            show=False
        )

        # 5. Radar chart
        self.plot_radar_chart(
            save_path=str(output_path / "radar_chart.png"),
            show=False
        )

        # 6. Entity type analysis
        print("\nAnalyzing entity type performance...")
        best_per_entity = self.find_best_model_per_entity()
        with open(output_path / "best_model_per_entity.json", 'w', encoding='utf-8') as f:
            json.dump(best_per_entity, f, ensure_ascii=False, indent=2)
        print(f"✓ Best model per entity type saved to: {output_path / 'best_model_per_entity.json'}")

        print(f"\n{'='*80}")
        print(f"✓ Full report generated at: {output_path}")
        print(f"{'='*80}")
        print(f"File list:")
        print(f"  - summary.csv                      # Summary table")
        print(f"  - summary.json                     # Summary data")
        print(f"  - f1_comparison.png                # F1 comparison")
        print(f"  - precision_comparison.png         # Precision comparison")
        print(f"  - recall_comparison.png            # Recall comparison")
        print(f"  - training_time_comparison.png     # Training time comparison")
        print(f"  - precision_recall_scatter.png     # Precision-Recall scatter plot")
        print(f"  - radar_chart.png                  # Radar chart")
        print(f"  - best_model_per_entity.json       # Best model per entity type")
        print(f"{'='*80}")


def main():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluation and visualization tool")
    parser.add_argument("--dir", required=True, help="Directory containing result.json files")
    parser.add_argument("--output", default="evaluation_report", help="Output directory")
    parser.add_argument("--pattern", default="*/result.json", help="File matching pattern")

    args = parser.parse_args()

    # Create evaluator
    evaluator = ResultEvaluator.from_directory(args.dir, args.pattern)

    # Generate full report
    evaluator.create_full_report(args.output)


if __name__ == "__main__":
    main()
