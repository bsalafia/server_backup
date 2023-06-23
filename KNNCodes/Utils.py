# ########################################################
# Alternative Convolution layers
# ########################################################

import keras.backend as K
import re


def get_numbered_name(name, prefix=None):
    """
    Rename the layer to append a number after the nase name for the layer
    :param name: Base name
    :param prefix: Prefix for the name of the layer
    :return:
    """
    if prefix is not None:
        name = prefix + '_' + name

    return '{}_{}'.format(name, K.get_uid(name))


def _to_snake_case(name):
    intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
    insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
    # If the class is private the name starts with "_" which is not secure
    # for creating scopes. We prefix the name with "private" in this case.
    if insecure[0] != '_':
        return insecure
    return 'private' + insecure


def get_numbered_layer_name(layer, prefix=None):
    """
    Get the default name of a layer by Keras standards
    :param layer: Layer
    :param prefix: Prefix for the name of the layer
    :return:
    """

    name = _to_snake_case(layer.__name__)

    if prefix is not None:
        name = prefix + '_' + name

    return get_numbered_name(name)

def query_tabular_data(data, query):
    """Query a tabular data structure like pandas DataFrame or Turi Create SFrame

    See the example for more details.

    Args:
        query: an array of query criteria
        data: data in a tabular format.

    Returns:
        rows that match query criteria

    Examples:
        Let's assume 'data' is a pandas DataFrame that contains some data. The following code will extract all the rows
        that are associated with {white OR black images} AND {have a device ID D1 OR D2} AND {have a frequency of F3}.

        query = {'image': [('white', '=='), ('black', '==')],
             'deviceID': [('D1', '=='), ('D2', '==')],
             'freq': [('F3', '==')]
             }
        match = query_dataframe(data, query=query)

    """
    all_keys = query.keys()

    curr_sframe = data

    query_syntax = "curr_sframe["
    counter = 0
    for curr_query_param in all_keys:
        new_key = True
        for param_values, op in query[curr_query_param]:
            if counter > 0 and new_key:
                query_syntax += "&"
            elif counter > 0:
                query_syntax += "|"
            if new_key:
                query_syntax += "("
            query_syntax += "(curr_sframe[\"{}\"] {} ".format(curr_query_param, op)
            if isinstance(param_values, str):
                query_syntax += "\"{}\")".format(param_values)
            else:
                query_syntax += "{})".format(param_values)
            new_key = False
            counter += 1
        query_syntax += ")"

    query_syntax += "]"
    selected = eval(query_syntax)
    return selected