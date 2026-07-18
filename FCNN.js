
function FCNN() {

    /////////////////////////////////////////////////////////////////////////////
                          ///////    Variables    ///////
    /////////////////////////////////////////////////////////////////////////////

    var w = window.innerWidth;
    var h = window.innerHeight;

    var svg = d3.select("#graph-container").append("svg").attr("xmlns", "http://www.w3.org/2000/svg");
    var g = svg.append("g");
    svg.style("cursor", "move");

    var edgeWidthProportional = true;
    var edgeWidth = 0.5;
    var weightedEdgeWidth = d3.scaleLinear().domain([0, 1]).range([0.3, edgeWidth * 5]);

    var edgeOpacityProportional = false;
    var edgeOpacity = 1.0
    var weightedEdgeOpacity = d3.scaleLinear().domain([0, 1]).range([0.15, 1]);

    var edgeColorProportional = false;
    var defaultEdgeColor = "#505050";
    var negativeEdgeColor = "#0000ff";
    var positiveEdgeColor = "#ff0000";

    var nodeDiameter = 20;
    var nodeColor = "#ffffff";
    var nodeBorderColor = "#333333";

    var betweenLayers = 160;

    var architecture = [8, 12, 8];
    var betweenNodesInLayer = [20, 20, 20];
    var graph = {};
    var layer_offsets = [];
    var largest_layer_width = 0;
    var nnDirection = 'right';
    var showLabels = true;
    var showArrowheads = false;
    var arrowheadStyle = "empty";
    var bezierCurves = false;
    var showWeightLabels = false;

    let sup_map = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'};
    let sup = (s) => Array.prototype.map.call(s, (d) => (d in sup_map && sup_map[d]) || d).join('');

    let textFn = (layer_index, layer_width) => ((layer_index === 0 ? "Input" : (layer_index === architecture.length-1 ? "Output" : "Hidden")) + " Layer ∈ ℝ" + sup(layer_width.toString()));
    var nominal_text_size = 12;
    var textWidth = 70;

    var marker = svg.append("svg:defs").append("svg:marker")
        .attr("id", "arrow")
        .attr("viewBox", "0 -5 10 10")
        .attr("markerWidth", 7)
        .attr("markerHeight", 7)
        .attr("orient", "auto");

    var arrowhead = marker.append("svg:path")
        .attr("d", "M0,-5L10,0L0,5")
        .style("stroke", defaultEdgeColor);

    var link = g.selectAll(".link");
    var hoverLink = g.selectAll(".link-hover");
    var node = g.selectAll(".node");
    var text = g.selectAll(".text");
    var weightLabel = g.selectAll(".weight-label");

    var tooltip = d3.select("body").append("div")
        .attr("id", "tooltip")
        .style("position", "absolute")
        .style("display", "none")
        .style("background", "rgba(255,255,255,0.95)")
        .style("border", "1px solid #ccc")
        .style("border-radius", "4px")
        .style("box-shadow", "0 1px 4px rgba(0,0,0,0.2)")
        .style("padding", "4px 8px")
        .style("font-size", "12px")
        .style("pointer-events", "none")
        .style("z-index", 2000);

    /////////////////////////////////////////////////////////////////////////////
                          ///////    Model    ///////
    /////////////////////////////////////////////////////////////////////////////

    var initScheme = 'xavier';
    var seed = 42;
    var weights = [];      // weights[layer][toNeuron][fromNeuron], layer = 1..L-1
    var biases = [];       // biases[layer][neuron], layer = 1..L-1
    var inputs = [];       // inputs[neuron], values of the input layer
    var activations = [];  // activations[layer], layer = 1..L-1
    var zs = [];           // zs[layer][neuron]
    var as = [];           // as[layer][neuron]; as[0] === inputs
    var maxAbsWeight = 1;

    var activationFunctions = {
        'linear':     z => z,
        'relu':       z => Math.max(0, z),
        'leaky-relu': z => (z >= 0 ? z : 0.01 * z),
        'sigmoid':    z => 1 / (1 + Math.exp(-z)),
        'tanh':       z => Math.tanh(z)
    };

    function mulberry32(a) {
        return function() {
            a |= 0; a = a + 0x6D2B79F5 | 0;
            var t = Math.imul(a ^ a >>> 15, 1 | a);
            t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
            return ((t ^ t >>> 14) >>> 0) / 4294967296;
        }
    }

    // Box-Muller; one sample per call so the draw order stays deterministic
    function gaussian(rng) {
        let u = 1 - rng();
        let v = rng();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    function initializeWeights({initScheme_=initScheme, seed_=seed}={}) {
        initScheme = initScheme_;
        seed = seed_;

        let rng = mulberry32(seed);

        weights = [];
        biases = [];
        for (let layer = 1; layer < architecture.length; layer++) {
            let fan_in = architecture[layer-1];
            let fan_out = architecture[layer];
            let sample;
            if      (initScheme === 'xavier') { let std = Math.sqrt(2 / (fan_in + fan_out)); sample = () => gaussian(rng) * std; }
            else if (initScheme === 'he')     { let std = Math.sqrt(2 / fan_in);             sample = () => gaussian(rng) * std; }
            else if (initScheme === 'normal') { sample = () => gaussian(rng) * 0.5; }
            else                              { sample = () => rng() * 2 - 1; }
            weights[layer] = range(fan_out).map(() => range(fan_in).map(() => sample()));
        }
        for (let layer = 1; layer < architecture.length; layer++) {
            biases[layer] = range(architecture[layer]).map(() => rng() * 0.2 - 0.1);
        }

        maxAbsWeight = Math.max(...flatten(weights.filter(x => x)).map(Math.abs), 1e-6);

        forward();
    }

    function ensureModel() {
        let L = architecture.length;

        if (activations.length !== L) {
            activations = architecture.map((_, l) => (l === L-1 ? 'linear' : 'relu'));
        }

        inputs = range(architecture[0] || 0).map(i => (typeof inputs[i] === 'number' && isFinite(inputs[i]) ? inputs[i] : 1.0));

        let shapeOk = L > 1 && weights.length === L &&
            architecture.every((width, l) => l === 0 ||
                (Array.isArray(weights[l]) && weights[l].length === width &&
                 weights[l].every(row => row.length === architecture[l-1])));

        if (!shapeOk) { initializeWeights(); }
        else { forward(); }
    }

    function forward() {
        zs = [];
        as = [];
        as[0] = inputs.slice();
        for (let layer = 1; layer < architecture.length; layer++) {
            zs[layer] = range(architecture[layer]).map(j =>
                weights[layer][j].reduce((acc, w_ji, i) => acc + w_ji * as[layer-1][i], 0) + biases[layer][j]);
            if (activations[layer] === 'softmax') {
                // softmax acts on the whole layer; shift by max(z) for numerical stability
                let maxZ = Math.max(...zs[layer]);
                let exps = zs[layer].map(z => Math.exp(z - maxZ));
                let sumExps = exps.reduce((acc, e) => acc + e, 0);
                as[layer] = exps.map(e => e / sumExps);
            } else {
                let f = activationFunctions[activations[layer]] || activationFunctions['linear'];
                as[layer] = zs[layer].map(f);
            }
        }
    }

    /////////////////////////////////////////////////////////////////////////////
                          ///////    Methods    ///////
    /////////////////////////////////////////////////////////////////////////////

    function redraw({architecture_=architecture,
                     showLabels_=showLabels,
                     bezierCurves_=bezierCurves,
                     }={}) {

        architecture = architecture_;
        showLabels = showLabels_;
        bezierCurves = bezierCurves_;

        ensureModel();

        graph.nodes = architecture.map((layer_width, layer_index) => range(layer_width).map(node_index => {return {'id':layer_index+'_'+node_index,'layer':layer_index,'node_index':node_index}}));
        graph.links = pairWise(graph.nodes).map((nodes) => nodes[0].map(left => nodes[1].map(right => {return {'id':left.id+'-'+right.id,
                                                                                                             'source':left.id,'target':right.id,
                                                                                                             'source_layer':left.layer,'source_index':left.node_index,
                                                                                                             'target_layer':right.layer,'target_index':right.node_index,
                                                                                                             'weight':weights[right.layer][right.node_index][left.node_index]}})));
        graph.nodes = flatten(graph.nodes);
        graph.links = flatten(graph.links).filter(l => l);

        label = architecture.map((layer_width, layer_index) => { return {'id':'layer_'+layer_index+'_label','layer':layer_index,'text':textFn(layer_index, layer_width)}});

        link = link.data(graph.links, d => d.id);
        link.exit().remove();
        link = link.enter()
                   .insert("path", ".link-hover")
                   .attr("class", "link")
                   .style("pointer-events", "none")
                   .merge(link);

        hoverLink = hoverLink.data(graph.links, d => d.id);
        hoverLink.exit().remove();
        hoverLink = hoverLink.enter()
                   .insert("path", ".node")
                   .attr("class", "link-hover")
                   .style("stroke", "#000")
                   .style("stroke-opacity", 0)
                   .style("fill", "none")
                   .style("pointer-events", "stroke")
                   .on("mouseover", link_mouseover)
                   .on("mousemove", move_tooltip)
                   .on("mouseout", hide_tooltip)
                   .merge(hoverLink);

        node = node.data(graph.nodes, d => d.id);
        node.exit().remove();
        node = node.enter()
                   .append("circle")
                   .attr("r", nodeDiameter/2)
                   .attr("class", "node")
                   .attr("id", function(d) { return d.id; })
                   .style("cursor", "pointer")
                   .on("click", node_clicked)
                   .on("mouseover", node_mouseover)
                   .on("mousemove", move_tooltip)
                   .on("mouseout", hide_tooltip)
                   .merge(node);

        weightLabel = weightLabel.data(showWeightLabels ? graph.links : [], d => d.id);
        weightLabel.exit().remove();
        weightLabel = weightLabel.enter()
                   .append("text")
                   .attr("class", "weight-label")
                   .attr("dy", "-2")
                   .style("font-size", "7px")
                   .style("pointer-events", "none")
                   .merge(weightLabel)
                   .text(d => d.weight.toFixed(2));

        text = text.data(label, d => d.id);
        text.exit().remove();
        text = text.enter()
                   .append("text")
                   .attr("class", "text")
                   .attr("dy", ".35em")
                   .style("font-size", nominal_text_size+"px")
                   .merge(text)
                   .text(function(d) { return (showLabels ? d.text : ""); });

        if (selectedNode && !graph.nodes.some(n => n.id === selectedNode.id)) { selectNode(null); }

        style();
    }

    function redistribute({betweenNodesInLayer_=betweenNodesInLayer,
                           betweenLayers_=betweenLayers,
                           nnDirection_=nnDirection,
                           bezierCurves_=bezierCurves}={}) {

        betweenNodesInLayer = betweenNodesInLayer_;
        betweenLayers = betweenLayers_;
        nnDirection = nnDirection_;
        bezierCurves = bezierCurves_;

        layer_widths = architecture.map((layer_width, i) => layer_width * nodeDiameter + (layer_width - 1) * betweenNodesInLayer[i])

        largest_layer_width = Math.max(...layer_widths);

        layer_offsets = layer_widths.map(layer_width => (largest_layer_width - layer_width) / 2);

        let indices_from_id = (id) => id.split('_').map(x => parseInt(x));

        let x = (layer, node_index) => layer * (betweenLayers + nodeDiameter) + w/2 - (betweenLayers * layer_offsets.length/3);
        let y = (layer, node_index) => layer_offsets[layer] + node_index * (nodeDiameter + betweenNodesInLayer[layer]) + h/2 - largest_layer_width/2;

        let xt = (layer, node_index) => layer_offsets[layer] + node_index * (nodeDiameter + betweenNodesInLayer[layer]) + w/2  - largest_layer_width/2;
        let yt = (layer, node_index) => layer * (betweenLayers + nodeDiameter) + h/2 - (betweenLayers * layer_offsets.length/3);

        if (nnDirection == 'up') { x = xt; y = yt; }

        node.attr('cx', function(d) { return x(d.layer, d.node_index); })
            .attr('cy', function(d) { return y(d.layer, d.node_index); });

        let pathFn;
        if(bezierCurves) {
            pathFn = (d) => {
                let source = [x(...indices_from_id(d.source)), y(...indices_from_id(d.source))];
                let target = [x(...indices_from_id(d.target)), y(...indices_from_id(d.target))];

                // control points
                let cp1 = [(source[0] + target[0]) / 2, source[1]];
                let cp2 = [(source[0] + target[0]) / 2, target[1]];

                return "M" + source[0] + "," + source[1]
                    + "C" + cp1[0] + "," + cp1[1]
                    + " " + cp2[0] + "," + cp2[1]
                    + " " + target[0] + "," + target[1];
            };
        } else {
            pathFn = (d) => "M" + x(...indices_from_id(d.source)) + "," +
                                  y(...indices_from_id(d.source)) + ", " +
                                  x(...indices_from_id(d.target)) + "," +
                                  y(...indices_from_id(d.target));
        }

        link.attr("d", pathFn);
        hoverLink.attr("d", pathFn);

        // spread labels along each edge (by source index) so they don't all pile up at the midpoint
        let labelT = (d) => 0.15 + 0.7 * ((d.source_index + 0.5) / architecture[d.source_layer]);
        weightLabel.attr("x", (d) => { let sx = x(d.source_layer, d.source_index), tx = x(d.target_layer, d.target_index); return sx + (tx - sx) * labelT(d); })
                   .attr("y", (d) => { let sy = y(d.source_layer, d.source_index), ty = y(d.target_layer, d.target_index); return sy + (ty - sy) * labelT(d); });

        text.attr("x", function(d) { return (nnDirection === 'right' ? x(d.layer, d.node_index) - textWidth/2 : w/2 + largest_layer_width/2 + 20 ); })
            .attr("y", function(d) { return (nnDirection === 'right' ? h/2 + largest_layer_width/2 + 20       : y(d.layer, d.node_index) ); });

    }

    function style({edgeWidthProportional_=edgeWidthProportional,
                    edgeWidth_=edgeWidth,
                    edgeOpacityProportional_=edgeOpacityProportional,
                    edgeOpacity_=edgeOpacity,
                    negativeEdgeColor_=negativeEdgeColor,
                    positiveEdgeColor_=positiveEdgeColor,
                    edgeColorProportional_=edgeColorProportional,
                    defaultEdgeColor_=defaultEdgeColor,
                    nodeDiameter_=nodeDiameter,
                    nodeColor_=nodeColor,
                    nodeBorderColor_=nodeBorderColor,
                    showArrowheads_=showArrowheads,
                    arrowheadStyle_=arrowheadStyle,
                    bezierCurves_=bezierCurves}={}) {
        // Edge Width
        edgeWidthProportional   = edgeWidthProportional_;
        edgeWidth               = edgeWidth_;
        weightedEdgeWidth       = d3.scaleLinear().domain([0, maxAbsWeight]).range([0.3, Math.max(edgeWidth * 5, 1)]);
        // Edge Opacity
        edgeOpacityProportional = edgeOpacityProportional_;
        edgeOpacity             = edgeOpacity_;
        weightedEdgeOpacity     = d3.scaleLinear().domain([0, maxAbsWeight]).range([0.15, 1]);
        // Edge Color
        defaultEdgeColor        = defaultEdgeColor_;
        edgeColorProportional   = edgeColorProportional_;
        negativeEdgeColor       = negativeEdgeColor_;
        positiveEdgeColor       = positiveEdgeColor_;
        // Node Styles
        nodeDiameter            = nodeDiameter_;
        nodeColor               = nodeColor_;
        nodeBorderColor         = nodeBorderColor_;
        // Arrowheads
        showArrowheads          = showArrowheads_;
        arrowheadStyle          = arrowheadStyle_;
        // Bezier curves
        bezierCurves            = bezierCurves_;

        link.style("stroke-width", linkWidth);

        link.style("stroke-opacity", linkOpacity);

        link.style("stroke", linkColor);

        link.style("fill", "none");

        link.attr('marker-end', showArrowheads ? "url(#arrow)" : '');
        marker.attr('refX', nodeDiameter*1.4 + 12);
        arrowhead.style("fill", arrowheadStyle === 'empty' ? "none" : defaultEdgeColor);

        hoverLink.style("stroke-width", function(d) { return Math.max(9, linkWidth(d)); });

        weightLabel.style("fill", linkColor);

        node.attr("r", nodeDiameter/2);
        node.style("fill", nodeColor);
        node.style("stroke", nodeBorderColor);

        applyFocus();

    }

    function linkWidth(d) {
        if (edgeWidthProportional) { return weightedEdgeWidth(Math.abs(d.weight)); } else { return edgeWidth; }
    }

    function linkOpacity(d) {
        if (edgeOpacityProportional) { return weightedEdgeOpacity(Math.abs(d.weight)); } else { return edgeOpacity; }
    }

    function linkColor(d) {
        if (edgeColorProportional) { return (d.weight >= 0 ? positiveEdgeColor : negativeEdgeColor); } else { return defaultEdgeColor; }
    }

    /////////////////////////////////////////////////////////////////////////////
                          ///////    Tooltip    ///////
    /////////////////////////////////////////////////////////////////////////////

    let fmt = (v) => (typeof v === 'number' && isFinite(v) ? v.toFixed(4) : String(v));

    function link_mouseover(d) {
        tooltip.style("display", "block")
               .html("w (" + d.source + " → " + d.target + ") = <span style='font-weight:bold'>" + fmt(d.weight) + "</span>");
        move_tooltip();
    }

    function node_mouseover(d) {
        let value = (d.layer === 0) ? inputs[d.node_index] : (as[d.layer] ? as[d.layer][d.node_index] : undefined);
        let name = (d.layer === 0) ? "x" : "a";
        tooltip.style("display", "block")
               .html(name + " (" + d.id + ") = <span style='font-weight:bold'>" + fmt(value) + "</span> → click for details");
        move_tooltip();
    }

    function move_tooltip() {
        tooltip.style("left", (d3.event.pageX + 14) + "px")
               .style("top", (d3.event.pageY - 12) + "px");
    }

    function hide_tooltip() {
        tooltip.style("display", "none");
    }

    /////////////////////////////////////////////////////////////////////////////
                          ///////    Selection & Focus    ///////
    /////////////////////////////////////////////////////////////////////////////

    var selectedNode = null;
    var nodeSelectCallback = null;

    function node_clicked(d) {
        d3.event.stopPropagation();
        selectNode(selectedNode && selectedNode.id === d.id ? null : d);
    }

    function selectNode(d) {
        selectedNode = d;
        applyFocus();
        if (nodeSelectCallback) { nodeSelectCallback(d); }
    }

    function applyFocus() {
        if (selectedNode) {
            let sel = selectedNode;
            // input neurons highlight their outgoing edges; all others their incoming edges
            let relevantLink = (o) => (sel.layer === 0 ? o.source === sel.id : o.target === sel.id);
            let relevantNode = (o) => (o.id === sel.id || (sel.layer === 0 ? o.layer === 1 : o.layer === sel.layer - 1));
            node.style("opacity", (o) => relevantNode(o) ? 1 : 0.45)
                .style("stroke-width", (o) => o.id === sel.id ? 3 : 1);
            link.style("opacity", (o) => relevantLink(o) ? 1 : 0.15);
            weightLabel.style("opacity", (o) => relevantLink(o) ? 1 : 0.15);
        } else {
            node.style("opacity", 1).style("stroke-width", 1);
            link.style("opacity", 1).style("stroke-opacity", linkOpacity);
            weightLabel.style("opacity", 1);
        }
    }

    svg.on("click", function() { if (selectedNode) { selectNode(null); } });

    /////////////////////////////////////////////////////////////////////////////
                          ///////    Zoom & Resize   ///////
    /////////////////////////////////////////////////////////////////////////////

    svg.call(d3.zoom()
               .scaleExtent([1 / 2, 8])
               .on("zoom", zoomed));

    function zoomed() { g.attr("transform", d3.event.transform); }

    function resize() {
        w = window.innerWidth;
        h = window.innerHeight;
        svg.attr("width", w).attr("height", h);
    }

    d3.select(window).on("resize", resize)

    resize();

    /////////////////////////////////////////////////////////////////////////////
                          ///////    Return    ///////
    /////////////////////////////////////////////////////////////////////////////

    return {
        'redraw'              : redraw,
        'redistribute'        : redistribute,
        'style'               : style,

        'initializeWeights'   : initializeWeights,
        'forward'             : forward,
        'setInput'            : function(i, v) { inputs[i] = v; forward(); },
        'setActivation'       : function(layer, name) { activations[layer] = name; forward(); },
        'setShowWeightLabels' : function(v) { showWeightLabels = v; redraw(); redistribute(); },
        'selectNode'          : selectNode,
        'onNodeSelect'        : function(cb) { nodeSelectCallback = cb; },
        'getModel'            : function() { return {'architecture': architecture, 'weights': weights, 'biases': biases,
                                                     'inputs': inputs, 'activations': activations, 'zs': zs, 'as': as,
                                                     'initScheme': initScheme, 'seed': seed}; },
        'activationFunctions' : activationFunctions,

        'graph'               : graph
    }

}
