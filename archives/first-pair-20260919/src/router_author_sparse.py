"""Known sparse recipes on the verified author architecture, with identical main initialization."""
import torch
from router_author_control import make_model
from frozen_routing import TokenIndexer
from router_recipe_baselines import RecipeRouter,RECIPES
from router_pressure_models import ControlRouter
from zoology_entry import set_determinism

METHODS=(*RECIPES,'fixed_values6')

def initial_indexers():
    set_determinism(2026091620)
    return torch.nn.ModuleList([TokenIndexer(128,16) for _ in range(2)]).state_dict()

def build(config,main_state,index_state,method,device):
    assert method in METHODS
    m=make_model(config,main_state,device)
    ix=torch.nn.ModuleList([TokenIndexer(128,16) for _ in range(2)] if method in RECIPES else []).to(device)
    if method in RECIPES:
        ix.load_state_dict(index_state);route=RecipeRouter(m,ix,method)
    else:route=ControlRouter(m,method)
    return m,ix,route
