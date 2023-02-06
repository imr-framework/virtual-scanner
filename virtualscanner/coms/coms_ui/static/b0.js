let socketio = io();
$('#run-halbach').on('click',()=>{
    // Get parameters and send them through socket 
    socketio.emit('B0 run Halbach',{'innerRadii':$("#innerRadii").val(),
                                     'innerNum':$('#innerNum').val(),
                                     'numRings':$('#numRings').val(),
                                     'ringSep':$('#ringSep').val(),
                                     'dsv':$('#dsv').val()})
    $("#run-halbach").attr('disabled',true);
    $('#run-halbach-spinner').removeClass('d-none');
    $("#output-space").text("The program is running...");


})

socketio.on('B0 print Halbach program output',(payload)=>{
    $("#output-space").text(payload['output']);
})

    // $('#t1-map').attr('disabled',true);
    // $('#t1-map-text').text("Calculating...");
    // $('#t1-map-spinner').removeClass('d-none');

socketio.on('Update B0 plot',(data)=>{
    console.log('Trying to update B0 plot...')
    graphData = JSON.parse(data['graphData']);
    Plotly.newPlot('halbach-chart',graphData);
    $("#run-halbach").attr('disabled',false);
    $('#run-halbach-spinner').addClass('d-none');
})