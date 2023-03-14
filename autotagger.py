import os
import json

from collections import OrderedDict
from pathlib import Path
from glob import glob
from PIL import Image, UnidentifiedImageError
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import requests
import tqdm
import multiprocessing as mp
from multiprocessing.pool import ThreadPool as Pool

from tagger import format, utils, dbimutils
from tagger.utils import split_str
from tagger.interrogator import Interrogator


DEFAULT_THRESHOLD = 0.35
DEFAULT_FILENAME_FORMAT = "[name].[output_extension]"

# kaomoji from WD 1.4 tagger csv. thanks, Meow-San#5400!
DEFAULT_REPLACE_UNDERSCORE_EXCLUDES = "0_0, (o)_(o), +_+, +_-, ._., <o>_<o>, <|>_<|>, =_=, >_<, 3_3, 6_9, >_o, @_@, ^_^, o_o, u_u, x_x, |_|, ||_||"


@dataclass
class InterrogateBatchOptions:
    input_glob: str
    input_recursive: bool
    output_dir: str
    output_filename_format: str
    output_action_on_conflict: str
    remove_duplicated_tag: bool
    output_save_json: bool


def unload_interrogators():
    unloaded_models = 0

    for i in utils.interrogators.values():
        if i.unload():
            unloaded_models = unloaded_models + 1

    return [f'Successfully unload {unloaded_models} model(s)']


def interrogate_single(
    image: Image,

    interrogator: str,
    threshold: float=DEFAULT_THRESHOLD,
    additional_tags: str="",
    exclude_tags: str="",
    sort_by_alphabetical_order: bool=False,
    add_confident_as_weight: bool=False,
    replace_underscore: bool=True,
    replace_underscore_excludes: str=DEFAULT_REPLACE_UNDERSCORE_EXCLUDES,
    escape_tag: bool=True,

    unload_model_after_running: bool=True
):
    if interrogator not in utils.interrogators:
        raise Exception(f"'{interrogator}' is not a valid interrogator")

    interrogator: Interrogator = utils.interrogators[interrogator]

    postprocess_opts = (
        threshold,
        split_str(additional_tags),
        split_str(exclude_tags),
        sort_by_alphabetical_order,
        add_confident_as_weight,
        replace_underscore,
        split_str(replace_underscore_excludes),
        escape_tag
    )

    ratings, tags = interrogator.interrogate(image)
    processed_tags = Interrogator.postprocess_tags(
        tags,
        *postprocess_opts
    )

    if unload_model_after_running:
        interrogator.unload()

    return (', '.join(processed_tags), ratings, tags)
