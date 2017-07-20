var width = 960,
    height = 720;

var zoom = d3.behavior.zoom()
    .scaleExtent([0.1, 10])
    .on("zoom", zoomed);

var svg = d3.select("#network_container").append("svg")
    .attr("width", width)
    .attr("height", height)
    .append("g")
    .call(zoom);

var rect = svg.append("rect")
    .attr("width", width)
    .attr("height", height)
    .attr("fill", "none")
    .style("pointer-events", "all");

var container = svg.append("g");

d3.json("network.json", function(network) {
    container.selectAll("g.layers")
             .data(network.layers)
             .enter()
             .append("g")
             .attr("class", "layer")
                 .selectAll("circle.node")
                 .data(function(layer, index) {
                     next_layer = network.layers.find(function(item, i) { return item.in === layer.name; });
                     if (next_layer) {
                         return next_layer.weights.map(function(e, i) { return [e, index]; })
                     } else {
                         return []
                     }
                 })
                 .enter()
                 .append("svg:circle")
                 .attr("class", "node")
                 .attr("cx", function(d, i, j) { return 100 + j * 400; })
                 .attr("cy", function(d, i) { return 20 + i * 22; })
                 .attr("r", "10px")
                     .selectAll("line.connection")
                     .data(function(d) { data = d[0].map(function(e, i) { return [e, d[1]]; }); console.log(data); return data; });
                     .enter()
                     .append("svg:line")
                     .attr("class", "connection")
                     .attr("x1", function(d) { return 100 + d[] * 400; })
                     .attr("y1", function(d) { return 20 +  // CONTINUE HERE
                     
});

function zoomed() {
    container.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
}
