"""Bio-inspired navigation and advanced SNN modules"""
from .navigation_cells import (
    HeadDirectionCell,
    GridCell,
    PlaceCell,
    PathIntegrator,
    BioInspiredNavigation,
    BioInspiredPolicy,
    STDPPlasticity,
    PredictiveCodingLayer,
    MultiTimescaleSNN
)
from .meta_plasticity import (
    LearnableSTDP,
    PlasticityRuleNetwork,
    MetaPlasticityOptimizer,
    MetaPlasticitySNN
)
from .neuromodulation import (
    Neuromodulator,
    GatedPlasticity,
    MetaPlasticityController,
    NeuromodulatedSNN,
    AdditiveNeuromodulatedLinear
)
from .attention_sensing import (
    SaliencyDetector,
    DynamicSensorArray,
    AttentionDrivenSensing,
    EventBasedAttention,
    HierarchicalAttention
)
from .world_model import (
    SNNTransitionModel,
    SNNRewardModel,
    PlanningHorizon,
    WorldModelSNN
)
from .continual_learning import (
    ElasticWeightConsolidation,
    MemoryReplay,
    ProgressiveNeuralNetworks,
    GradientProjection,
    ContinualLearningSNN,
    continual_learning_loop,
    evaluate_task
)

__all__ = [
    # Navigation
    'HeadDirectionCell', 'GridCell', 'PlaceCell', 'PathIntegrator',
    'BioInspiredNavigation', 'BioInspiredPolicy',
    # Meta-plasticity
    'LearnableSTDP', 'PlasticityRuleNetwork', 'MetaPlasticityOptimizer', 'MetaPlasticitySNN',
    # Neuromodulation
    'Neuromodulator', 'GatedPlasticity', 'MetaPlasticityController',
    'NeuromodulatedSNN', 'AdditiveNeuromodulatedLinear',
    # Attention
    'SaliencyDetector', 'DynamicSensorArray', 'AttentionDrivenSensing',
    'EventBasedAttention', 'HierarchicalAttention',
    # World model
    'SNNTransitionModel', 'SNNRewardModel', 'PlanningHorizon', 'WorldModelSNN',
    # Continual learning
    'ElasticWeightConsolidation', 'MemoryReplay', 'ProgressiveNeuralNetworks',
    'GradientProjection', 'ContinualLearningSNN', 'continual_learning_loop', 'evaluate_task',
]
