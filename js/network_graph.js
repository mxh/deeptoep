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

    new_layer.nodes = Array.apply(null, Array(new_layer.size))
                           .map(function(val) { return {"connections": [], "layer": layer.name, "input": 0}; });
    new_layer.nodes.forEach(function(node, idx) { node.idx = idx; });

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
    return 20 + node.idx * 22;
}

function get_connection_x1_location(connection, network)
{
    return 100 + connection.input_layer_level * 800;
}

function get_connection_y1_location(connection, network)
{
    return 20 + connection.input_node_idx * 22;
}

function get_connection_x2_location(connection, network)
{
    return 100 + connection.output_layer_level * 800;
}

function get_connection_y2_location(connection, network)
{
    return 20 + connection.output_node_idx * 22;
}

d3.json("network.json", function(network) {
    orig_network = network;
    full_network = convert_compact_to_full_network(network);
    update_network_viz();
});

function update_network_viz()
{
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
                                        .attr("r", "10px");

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
                                                                                                           + connection.output_layer_idx; })
                                                   .attr("x1", function(connection) { return get_connection_x1_location(connection); })
                                                   .attr("y1", function(connection) { return get_connection_y1_location(connection); })
                                                   .attr("x2", function(connection) { return get_connection_x2_location(connection); })
                                                   .attr("y2", function(connection) { return get_connection_y2_location(connection); })
                                                   .style("visibility", function(node) { if (node.weight < -0.5) { return "visible"; } else { return "hidden"; } });
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

function propagate_input(node)
{
    node.connections.forEach(function(connection) {
        get_layer_by_name(full_network, connection.output_layer).nodes[connection.output_node_idx].input +=
            connection.weight * node.input;
    }); // CONTINUE HERE
    asdfasdf
}

function update_network_input(state)
{
    var values = state.split('').map(Number);
    var input_layer = get_layer_by_name(full_network, 'input');

    reset_inputs();

    input_layer.nodes.forEach(function(node) {
        node.input = values[node.idx];
        propagate_input(node);
    });
}

$("#state_input").on("keypress", function(e) {
    if (e.which === 13)
    {
        $(this).attr("disabled", "disabled");

        update_network_input($(this).val());
        //update_network_viz();

        $(this).removeAttr("disabled");
    }
});
