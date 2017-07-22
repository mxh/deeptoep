var width = window.innerWidth,
    height = window.innerHeight;

var zoom = d3.behavior.zoom()
    .scaleExtent([0.1, 10])
    .on("zoom", zoomed);

var svg = d3.select("#network_container").append("svg")
    //.attr("width", width)
    //.attr("height", height)
    .append("g")
    .call(zoom);

var rect = svg.append("rect")
    .attr("width", width)
    .attr("height", height)
    .attr("fill", "none")
    .style("pointer-events", "all");

var container = svg.append("g");

function convert_compact_to_full_layer(layer, layer_idx, network)
{
    var new_layer = Object();
    new_layer.name = layer.name;
    new_layer.size = layer.size;
    new_layer.in = layer.in;
    new_layer.relu = layer.relu;
    new_layer.y_offset = layer.y_offset;
    console.log(layer);

    new_layer.nodes = Array.apply(null, Array(new_layer.size))
                           .map(function(val) { return {"connections": [], "layer": layer.name, "input": 0}; });
    new_layer.nodes.forEach(function(node, idx) { node.idx = idx; });
    new_layer.biases = layer.biases

    if ("node_labels" in layer) {
        layer.node_labels.forEach(function(label, node_idx) {
            new_layer.nodes[node_idx].label = label;
        });
    }

    network.layers.filter(function(output_layer) { return output_layer.in === layer.name; }) // for each output layer
                  .forEach(function(output_layer) {
                       var layer_view;
                       if ("start" in output_layer)
                       {
                           layer_view = new_layer.nodes.slice(output_layer.start, output_layer.end);
                       } else {
                           layer_view = new_layer.nodes;
                       }
                       
                       // for each node in each output layer, we add a connection for each node in the input
                       layer_view.forEach(function(node, input_node_idx) {
                           console.log(output_layer.name);
                           for (output_node_idx = 0; output_node_idx < output_layer.size; output_node_idx++)
                           {
                               node.connections.push({"input_layer": layer.name, "input_layer_level": layer.level, "output_layer": output_layer.name, "output_layer_level": output_layer.level, "input_node_idx": node.idx, "output_node_idx": output_node_idx, "weight": output_layer["weights"][input_node_idx][output_node_idx]});
                           }
                       });
                   });

    return new_layer;
}

function convert_compact_to_full_network(network)
{
    var full_network = Object();
    full_network.layers = [];

    network.layers.forEach(function(layer, layer_idx) {
        full_layer = convert_compact_to_full_layer(layer, layer_idx, network);
        full_network.layers.push(full_layer);
    });

    return full_network;
}

function get_layer_by_name(network, name)
{
    return network.layers.find(function(layer, idx) { return layer.name === name; });
}

function get_node_x_location(node, network)
{
    return 100 + get_layer_by_name(network, node.layer).level * 800;
}

function get_node_y_location(node, network)
{
    var y_offset = get_layer_by_name(network, node.layer).y_offset;
    return 20 + node.idx * 22 + y_offset * 44;
}

function get_connection_x1_location(connection, network)
{
    return 100 + connection.input_layer_level * 800;
}

function get_connection_y1_location(connection, network)
{
    var y_offset = get_layer_by_name(network, connection.input_layer).y_offset;
    return 20 + connection.input_node_idx * 22 + y_offset * 44;
}

function get_connection_x2_location(connection, network)
{
    return 100 + connection.output_layer_level * 800;
}

function get_connection_y2_location(connection, network)
{
    var y_offset = get_layer_by_name(network, connection.output_layer).y_offset;
    return 20 + connection.output_node_idx * 22 + y_offset * 44;
}

d3.json("network.json", function(network) {
    orig_network = network;
    full_network = convert_compact_to_full_network(network);
    update_network_viz();
});

activation_color_ramp = d3.scale.linear().domain([-1, 0, 1]).range(["red", "white", "green"]);

selected_node = undefined;

function handle_node_click(node)
{
    if (selected_node === node)
    {
        selected_node = undefined;
    } else {
        selected_node = undefined;
        update_selected_node_viz();
        selected_node = node;
    }
    update_selected_node_viz();
}

function select_important_connections_recursive(selected_node, n)
{
    // we want to only show the connections that are in the top-3 contributing connections for the selected node
    var connecting_layer = get_layer_by_name(full_network, get_layer_by_name(full_network, selected_node.layer).in);

    if (connecting_layer == undefined) return;

    var all_connections = [];
    connecting_layer.nodes.forEach(function(node) {
        node.connections.filter(function(connection) { return connection.output_node_idx === selected_node.idx; })
                        .forEach(function(connection) { all_connections.push({"connection": connection, "contribution": connection.weight * node.input}); });
    });

    all_connections.sort(function(connection_a, connection_b) {
        if (Math.abs(connection_a.contribution) < Math.abs(connection_b.contribution)) return 1;
        if (Math.abs(connection_a.contribution) > Math.abs(connection_b.contribution)) return -1;
        return 0;
    });

    for (var top_idx = 0; top_idx < 5; top_idx++)
    {
        var selector = "line#connection_" + all_connections[top_idx]["connection"].input_layer + "_"
                                          + all_connections[top_idx]["connection"].input_node_idx + "_"
                                          + all_connections[top_idx]["connection"].output_layer + "_"
                                          + all_connections[top_idx]["connection"].output_node_idx;
        console.log(top_idx);
        console.log(selector);
        container.selectAll(selector)
                 .classed("unselected", false)
                 .classed("selected", true);

        select_important_connections_recursive(get_layer_by_name(full_network, all_connections[top_idx]["connection"].input_layer).nodes[all_connections[top_idx]["connection"].input_node_idx], n);
    }
}

function update_selected_node_viz()
{
    if (selected_node)
    {
        container.selectAll("line.connection").classed("unselected", true);

        select_important_connections_recursive(selected_node, 3);
        
    } else {
        container.selectAll("line.connection").classed("unselected", false).classed("selected", false);
    }
}

function update_network_viz()
{
    container.html("")
    var per_layer = container.selectAll("g.layer")
                  .data(full_network.layers) // first, we iterate through the layers of our network
                  .enter();

    var per_node_group = per_layer.append("g") // for each layer, we have a group in the svg
                 .attr("class", "layer")
                 .selectAll("g.node")
                 .data(function(layer) { return layer.nodes; })
                 .enter()
                 .append("g")
                 .attr("class", "node")
                 .attr("id", function(d) { return "node_id_" + d.idx; });

    var per_node_circle = per_node_group.append("svg:circle")
                                        .attr("class", "node_circle")
                                        .attr("id", function(d) { return "node_circle_" + d.idx; })
                                        .attr("cx", function(d) { return get_node_x_location(d, orig_network); })
                                        .attr("cy", function(d) { return get_node_y_location(d, orig_network); })
                                        .style("fill", function(d) { return activation_color_ramp(d.input); }) 
                                        .attr("r", "10px")
                                        .on("click", handle_node_click);

    per_node_group.filter(function(node) { return node.label == "Stake"; })
                   .append("svg:text")
                   .attr("x", function(node) { console.log(node); return get_node_x_location(node, orig_network) - 2.5; })
                   .attr("y", function(node) { return get_node_y_location(node, orig_network) + 3; })
                   .text(function(node) { return node.input.toString(); });

    var per_node_label = per_node_group.filter(function(node) { return "label" in node; })
                                       .append("svg:text")
                                       .attr("text-anchor", "end")
                                       .attr("x", function(node) { return get_node_x_location(node, orig_network) - 20; })
                                       .attr("y", function(node) { return get_node_y_location(node, orig_network) + 3; })
                                       .text(function(node) { return node.label; });

    var per_node_score = per_node_group.append("svg:text")
                                       .attr("x", function(node) { return get_node_x_location(node, orig_network) + 20; })
                                       .attr("y", function(node) { return get_node_y_location(node, orig_network) + 3; })
                                       .text(function(node) { return node.input; });
                                

    var per_node_connections_group = per_node_group.append("g")
                                                   .attr("class", "connection_group");

    var per_connection = per_node_connections_group.selectAll("line.connection")
                                                   .data(function(node) { return node.connections; })
                                                   .enter()
                                                   .append("svg:line")
                                                   .attr("class", "connection")
                                                   .attr("id", function(connection) { return "connection_" + connection.input_layer + "_"
                                                                                                           + connection.input_node_idx + "_"
                                                                                                           + connection.output_layer + "_"
                                                                                                           + connection.output_node_idx; })
                                                   .attr("x1", function(connection) { return get_connection_x1_location(connection, full_network); })
                                                   .attr("y1", function(connection) { return get_connection_y1_location(connection, full_network); })
                                                   .attr("x2", function(connection) { return get_connection_x2_location(connection, full_network); })
                                                   .attr("y2", function(connection) { return get_connection_y2_location(connection, full_network); })
                                                   .style("visibility", function(connection) { if (Math.abs(connection.weight * get_layer_by_name(full_network, connection.input_layer).nodes[connection.input_node_idx].input) > 0.25) { return "visible"; } else { return "hidden"; } })
                                                   .style("stroke", function(connection) { if (connection.weight * get_layer_by_name(full_network, connection.input_layer).nodes[connection.input_node_idx].input > 0) { return "#00ff00"; } else { return "ff0000"; } });
}

function zoomed() {
    container.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
}

function reset_inputs(network) {
    network.layers.forEach(function(layer) {
        layer.nodes.forEach(function(node) {
            node.input = 0;
        });
    });
}

function propagate_input(layer, network)
{
    if ("biases" in layer && layer.biases)
    {
        console.log(layer.biases);
        layer.nodes.forEach(function(node) { node.input += layer.biases[node.idx]; });
    }

    if (layer.relu > 0)
    {
        console.log("relu-ing...");
        layer.nodes.forEach(function(node) { node.input = Math.max(0, node.input); });
    }


    layer.nodes.forEach(function(node) {
        node.connections.forEach(function(connection) {
            get_layer_by_name(full_network, connection.output_layer).nodes[connection.output_node_idx].input +=
                connection.weight * node.input;
        });
    });

    next_layers = network.layers.filter(function(next_layer) { return next_layer.in === layer.name; });
    next_layers.forEach(function(next_layer) { propagate_input(next_layer, network); })
}

function update_network_input(state)
{
    var values = state.split('').map(Number);
    var input_layer = get_layer_by_name(full_network, 'input');

    reset_inputs(full_network);

    input_layer.nodes.forEach(function(node) {
        node.input = values[node.idx];
    });

    propagate_input(input_layer, full_network);
}

$("#state_input").on("keypress", function(e) {
    if (e.which === 13)
    {
        $(this).attr("disabled", "disabled");

        update_network_input($(this).val());
        update_network_viz();

        $(this).removeAttr("disabled");
    }
});
