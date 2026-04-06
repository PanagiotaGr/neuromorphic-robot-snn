"""Multi-agent coordination and swarm intelligence"""
from .coordination import (
    CommunicationChannel,
    ConsensusModule,
    RoleAllocation,
    MultiAgentSNN,
    emergent_flocking
)

__all__ = [
    'CommunicationChannel',
    'ConsensusModule',
    'RoleAllocation',
    'MultiAgentSNN',
    'emergent_flocking'
]
