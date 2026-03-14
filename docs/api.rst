API reference
=============

Covariance
----------

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   simulacra.AR1Covariance
   simulacra.IsotropicCovariance
   simulacra.LKJCovariance
   simulacra.random_effects_covariance
   simulacra.residual_covariance

States
------

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   simulacra.PredictorState
   simulacra.EventTimeState
   simulacra.CensoredState
   simulacra.ObservedState
   simulacra.SurvivalState
   simulacra.TokenizedState
   simulacra.EventProcessState
   simulacra.CompetingRisksState
   simulacra.RiskIndicatorState
   simulacra.DiscreteRiskState

Predictor
---------

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   simulacra.linear_predictor
   simulacra.random_effects
   simulacra.activation
   simulacra.linear
   simulacra.mlp
   simulacra.observation_time

Response distributions
----------------------

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   simulacra.gaussian
   simulacra.poisson
   simulacra.bernoulli
   simulacra.categorical
   simulacra.ordinal
   simulacra.tokens

Survival
--------

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   simulacra.event_time
   simulacra.censor_time
   simulacra.mixture_cure_censoring
   simulacra.survival_indicators

Competing risks
---------------

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   simulacra.competing_risks
   simulacra.independent_events
   simulacra.risk_indicators
   simulacra.multi_event
   simulacra.discretize_risk

Simulation steps
----------------

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   simulacra.Simulation
   simulacra.ResponseStep
   simulacra.DiscreteResponseStep
   simulacra.TokenizedStep
   simulacra.EventTimeStep
   simulacra.CensoredStep
   simulacra.SurvivalStep
   simulacra.CompetingRisksStep
   simulacra.IndependentEventsStep
   simulacra.RiskIndicatorStep
   simulacra.DiscreteRiskStep
   simulacra.DiscreteEventTimeStep
