import collections
import json
import subprocess


def show_graph(op):
    source = render_graph(op)
    return show_dot(source)


def show_dot(s):
    from IPython.display import display_png, Image
    
    result = subprocess.run(['dot', '-Tpng'], input=s.encode('utf8'), stdout=subprocess.PIPE, check=True)
        
    img = Image(data=result.stdout, format='png')
    display_png(img)


def render_graph(op):
    visited = set()
    active = {op}
    
    nodes = collections.OrderedDict()
    graph = collections.OrderedDict()
    
    while active:
        current = active.pop()
        
        if current in visited:
            continue
        visited.add(current)
        
        if isinstance(current, (tf.Tensor, tf.Variable)):
            nodes[current.name] = '[label={}]'.format(json.dumps('{}: {}'.format(current.name, current.shape)))
            graph.setdefault(current.name, []).append(current.op.name)
            active.add(current.op)
        
        elif isinstance(current, tf.Operation):
            nodes[current.name] = '[label={}, shape=box]'.format(json.dumps(current.name))
            graph.setdefault(current.name, []).extend(inp.name for inp in current.inputs)
            active.update(current.inputs)
            
        else:
            raise ValueError('unknown node type: {}'.format(type(current)))
    
    source = []
    
    source.append('digraph {')
    source.extend(
        '{}{};'.format(json.dumps(node), desc)
        for node, desc in nodes.items()
    )
    
    source.extend(
        '{} -> {};'.format(json.dumps(src), json.dumps(dst))
        for dst, inputs in graph.items()
        for src in inputs
    )
    
    source.append('}')
    
    return '\n'.join(source)
