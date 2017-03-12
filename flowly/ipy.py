"""Helper functions to simplify working with ipython.

Relevant documentation:

* https://github.com/ipython/ipython/wiki/Dev:-Javascript-Events
* https://github.com/jupyter/notebook/blob/master/docs/source/comms.rst

"""
from __future__ import print_function, division, absolute_import

import base64
import io
import json

exec_js_comm = None


def init(components={'comms', 'toc'}):
    """Execute javascript initializations inside the notebook.

    :param Set[str] components:
        Which components to initialize.
        Currently the following components are known:

        * `comms`: to allow executing javascript code inside the notebook
        * `toc`:  to add a table of contents to the notebook.
    """
    if 'comms' in components:
        add_comms()

    if 'toc' in components:
        add_toc()


def set_default_indent(num_spaces):
    """Only takes effect for new cells.
    """
    from IPython.display import Javascript, display_javascript
    display_javascript(Javascript('''
        (function() {
            "use strict";

            if(require('base/js/namespace').notebook._fully_loaded){
                run();
            }
            else{
                require('base/js/events').on('notebook_loaded.Notebook', run);
            }

            function run() {
                require("notebook/js/cell").Cell.options_default.cm_config.indentUnit = %s;
            }
        })()
    ''' % json.dumps(num_spaces)))


def printmk(*args, **kwargs):
    """Print markdown in ipython.
    """
    if not args:
        return

    fmt, args = args[0], args[1:]

    from IPython.display import display_markdown, Markdown
    display_markdown(Markdown(fmt.format(*args, **kwargs)))


def notify(message, tag=None):
    """Send a browser notification.

    :param str message:
        the message to show

    :param str tag:
        if given, message with the same tag will replace on another.
        This option may be useful to prevent cluttering up the user's screen.

    Requires a initialization of the `commms` component, via
    :func:`flowly.ipy.init`.
    """
    options = {}
    if tag is not None:
        options['tag'] = tag

    exec_javascript(
        'new Notification({message}, {options});',
        message=json.dumps(message),
        options=json.dumps(options),
    )


def download_csv(df, filename='data.csv', **to_csv_kwargs):
    """Trigger the download of a CSV representation of the passed dataframe.

    :param pandas.DataFrame:
        the dataframe to download

    :param filename:
        the filename that is used to save the file under

    :param to_csv_kwargs:
        keyword arguments that are passed verbatim to `df.to_csv`

    Requires a initialization of the `commms` component, via
    :func:`flowly.ipy.init`.
    """
    encoding = to_csv_kwargs.get('encoding', 'utf-8')

    fobj = io.StringIO()
    df.to_csv(fobj, **to_csv_kwargs)
    content = fobj.getvalue()

    url = u'data:text/csv,base64,' + base64.b64encode(content.encode(encoding)).decode('ascii')

    exec_javascript(
        '''
            var link = document.createElement("a");
            link.download = {filename};
            link.href = {url};
            link.click();
        ''',
        url=json.dumps(url),
        filename=json.dumps(filename),
    )


def add_comms():
    """Initialize the comms used for communication.
    """
    from IPython.display import display_javascript, Javascript
    display_javascript(Javascript(init_comms_javascript))


def exec_javascript(source, **kwargs):
    """Execute the passed javascript source inside the notebook environment.

    Requires a prior call to ``init_comms()``.
    """
    global exec_js_comm

    if exec_js_comm is None:
        from ipykernel.comm import Comm
        comm = Comm("flowly_exec_js", {})

    if kwargs:
        source = source.format(**kwargs)

    comm.send({"source": source})


init_comms_javascript = '''
(function() {
    window.flowly_exec_js = function(msg) {
        var payload = msg['content']['data'];
        eval('' + payload['source']);
    }

    function register_comm() {
        Jupyter.notebook.kernel.comm_manager.register_target('flowly_exec_js', function(comm, msg){
            comm.on_msg(window.flowly_exec_js);
        });

    }

    if(Jupyter.notebook.kernel) {
        register_comm();
    }
    else {
        $([IPython.events]).on("kernel_ready.Kernel", register_comm);
    }
})();
'''


def add_toc():  # pragma: no cover
    """Add a dynamic table of contents to an ipython notebook.

    Any heading element (h1, h2, ...) in the DOM will add a link to the table of
    contents. The easiest way to add the headers is via IPythons markdown cells.
    Any edits or deletes of these cells will automatically be reflected in the
    generated table of contents.

    Usage::

        from flowly.ipy import add_toc
        add_toc()
    """
    from IPython.display import display_javascript, Javascript
    display_javascript(Javascript(js_source))


js_source = '''
(function(){
    "use strict";

    var Jupyter = require('base/js/namespace');
    if(Jupyter.notebook._fully_loaded){
        run();
    }
    else{
        require('base/js/events').on('notebook_loaded.Notebook', run);
    }

    function run() {
        var div = ensureContainer()

        setupStyle();
        registerEvents()
        updateTOC();
    }

    function setupStyle() {
        var site = $('#site');
        var top = 50;
        if(site.length) {
            top = site.position().top + 5;
        }

        var style = ensureStyleSection();

        style.empty();

        style.append('#ipytoc {');
        style.append('  padding: 0.5em;');
        style.append('  top: ' + top + 'px;');
        style.append('  left: 5px;');
        style.append('  position: fixed;');
        style.append('  width: 1em;');
        style.append('  height: 20em;');
        style.append('  overflow: hidden;');
        style.append('  background-color: #fea;');
        style.append('  z-index: 10;');
        style.append('  opacity: 0.95;');
        style.append('}');

        style.append('#ipytoc:hover {');
        style.append('  width: 30em;');
        style.append('  height: auto;');
        style.append('  min-height: 20em;');
        style.append('  background-color: #fff;');
        style.append('}');

        style.append('#ipytoc #ipytoc-content {');
        style.append('  display: none;');
        style.append('}');

        style.append('#ipytoc:hover #ipytoc-content {');
        style.append('  display: block;');
        style.append('}');

        style.append('#ipytoc h1 {');
        style.append('  font-size: 1em;');
        style.append('  padding: 0px;');
        style.append('  margin: 0px;');
        style.append('}');

        style.append('#ipytoc ul, #ipytoc li {');
        style.append('  padding: 0.05em 0px;');
        style.append('  margin: 0px;');
        style.append('}');

        style.append('.ipytoc-h1 { margin-left: 0em; }');
        style.append('.ipytoc-h2 { margin-left: 0.3em; }');
        style.append('.ipytoc-h3 { margin-left: 0.6em; }');
        style.append('.ipytoc-h4 { margin-left: 0.9em; }');
        style.append('.ipytoc-h5 { margin-left: 1.2em; }');
        style.append('.ipytoc-h6 { margin-left: 1.5em; }');
    }

    function ensureStyleSection() {
        var style = $('#ipytoc-style');

        if(style.length) {
            return style;
        }

        style = $('<style id="ipytoc-style" type="text/css"></style>');
        $('head').append(style);

        return style;
    }

    function ensureContainer() {
        var div = $('#ipytoc-content');

        if(div.length) {
            return div;
        }

        div = $('<div id="ipytoc"></div>');
        $("body").append(div);

        var content = $('<div id="ipytoc-content"></div>')
        div.append(content);

        return content;
    }

    function updateTOC() {
        var div = ensureContainer();
        var ul = $('<ul></ul>');
        var h1 = $('<h1>Table of contents</h1>');

        div.empty();
        div.append(h1)
        div.append(ul);

        ul.empty();

        $('h1,h2,h3,h4,h5,h6').each(function(){
            var target_id = $(this).attr('id');
            var target_title = $(this).text();
            target_title = target_title.replace("\xB6", '');

            if(target_id == undefined) {
                return;
            }
            var className = 'ipytoc-' + this.tagName.toLowerCase();
            ul.append($('<li><a class="' + className + '"href="#' + target_id + '">' + target_title + '</a></li>'));
        })
    }

    function registerEvents() {
        $([IPython.events]).on("create.Cell", updateTOC);
        $([IPython.events]).on("delete.Cell", updateTOC);
        $([IPython.events]).on("rendered.MarkdownCell", updateTOC);
    }
})();
'''
