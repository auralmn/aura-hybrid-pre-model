from .neuron import ActivityState, MaturationStage


class NeuronStateMachine:
    def __init__(self, stage=MaturationStage.PROGENITOR, activity=ActivityState.RESTING):
        self.stage = stage
        self.activity = activity
        self.membrane_potential = 0.0
        self.refractory_timer = 0

    def advance_stage(self, signals, gene_expression):
        # Example transition logic for development
        if self.stage == MaturationStage.PROGENITOR and signals.get('Wnt', 0) > 0.7:
            self.stage = MaturationStage.MIGRATING
        elif self.stage == MaturationStage.MIGRATING and gene_expression.get('Neurogenin', 0) > 0.8:
            self.stage = MaturationStage.DIFFERENTIATED
        elif self.stage == MaturationStage.DIFFERENTIATED and signals.get('FGF2', 0) > 0.5:
            self.stage = MaturationStage.MYELINATED

    def update_activity(self, input_current):
        if self.activity == ActivityState.REFRACTORY:
            self.refractory_timer -= 1
            if self.refractory_timer <= 0:
                self.activity = ActivityState.RESTING
        else:
            self.membrane_potential += input_current
            if self.membrane_potential > 1.0:
                self.activity = ActivityState.FIRING
                self.membrane_potential = 0.0
                self.activity = ActivityState.REFRACTORY
                self.refractory_timer = 3  # example refractory duration
            else:
                self.activity = ActivityState.RESTING
