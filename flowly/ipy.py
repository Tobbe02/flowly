from __future__ import print_function, division, absolute_import


def add_toc():
    """Add a dynamic table of contents to an ipython notebook.

    Usage::

        from flowly.ipy import add_toc
        add_toc()
    """
    from IPython.display import display_javascript, Javascript
    display_javascript(Javascript(js_source))


js_source = '''
(function(){
    "use strict";
    var div = ensureContainer()

    setupStyle();
    registerEvents()
    updateTOC();

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
        style.append('  width: 15em;');
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

        if(style.length) {
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
