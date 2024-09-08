from cl_methods.generative_replay import GenerativeReplay
from cl_methods.generative_replay_disjoint_classifier_guidance import (
    GenerativeReplayDisjointClassifierGuidance,
)
from cl_methods.generative_replay_online import GenerativeReplayOnline


def get_cl_method(args):
    if args.cl_method == "generative_replay":
        return GenerativeReplay(args)
    if args.cl_method == "generative_replay_disjoint_classifier_guidance":
        return GenerativeReplayDisjointClassifierGuidance(args)
    if args.cl_method == "online":
        return GenerativeReplayOnline(args)
    assert False, "bad cl method!"
