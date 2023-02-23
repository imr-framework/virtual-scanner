let socketio = io();
$('#run-halbach').on('click',()=>{
    // Get parameters and send them through socket 
    socketio.emit('B0 run Halbach',{'innerRadii':$("#innerRadii").val(),
                                     'innerNum':$('#innerNum').val(),
                                     'numRings':$('#numRings').val(),
                                     'ringSep':$('#ringSep').val(),
                                     'dsv':$('#dsv').val(),
                                     'maxGen': $("#max-gen").val(),
                                     'resolution':$('#resolution').val()})
    $("#run-halbach").attr('disabled',true);
    $('#run-halbach-spinner').removeClass('d-none');
    $("#output-space").text("The program is running...");


})

socketio.on('B0 print Halbach program output',(payload)=>{
     $("#output-space").text(`Best vector: ${payload['best_vector']}`);
 })

    // $('#t1-map').attr('disabled',true);
    // $('#t1-map-text').text("Calculating...");
    // $('#t1-map-spinner').removeClass('d-none');

socketio.on('Update B0 plot',(data)=>{
    graphData = JSON.parse(data['graphData']);
    Plotly.newPlot('halbach-chart',graphData);
    $("#run-halbach").attr('disabled',false);
    $('#run-halbach-spinner').addClass('d-none');

    $("#indx").val(data['x']);
    $("#indy").val(data['y']);
    $("#indz").val(data['z']);

})



$("#show-3d").on('click',()=>{
    socketio.emit("Get 3D plot");
})


socketio.on('Update 3D plot',(data)=>{
    graphData = JSON.parse(data['graphData']);
    Plotly.newPlot('halbach-chart-3d',graphData);
})

$("#update-halbach-slices").on('click',()=>{
    // Ask backend to redeliver imagex
    socketio.emit("Update Halbach slices",{'x':$('#indx').val(),
                                           'y':$("#indy").val(),
                                           'z':$("#indz").val(),
                                            'dsv_display':parseFloat($("#dsv_display").val()),
                                            'dsv':parseFloat($("#dsv").val())});
})

$('input[name=mag-options]').on('click',(event)=>{
    console.log(event.target.id);
    socketio.emit('Update B0 session',{'opt-3d': event.target.id[4]})
})

//--Update session as parameters are changed--------
$("#dsv_display").on('change',()=>{
    socketio.emit('Update B0 session', {'dsv_display': parseFloat($('#dsv_display').val())});
})
$('#res_display').on('change',()=>{
    socketio.emit('Update B0 session',  {'res_display': parseFloat($('#res_display').val())});
})



//--------------END OF SESSION UPDATES------------------


$("#show-magnet").on('click',()=>{
    socketio.emit("Get rings plot");
})

socketio.on("Update rings plot",(data)=>{
    graphData = JSON.parse(data['graphData']);
    Plotly.newPlot('rings-chart-3d',graphData)
})

// Save / load / mode existing simulated data
$('#b0-save').on("click",()=>{
    socketio.emit("Save B0 session to data");
    $("#b0-save").attr('disabled',true);
    $('#b0-save-spinner').removeClass('d-none');
})
socketio.on('B0 session saved',()=>{
    $("#output-space").text('B0 session data saved to halbach.mat');
    $("#b0-save").attr('disabled',false);
    $('#b0-save-spinner').addClass('d-none');
})

$('#b0-load').on("click",()=> {
    socketio.emit("Load B0 session from data",{'type':$('#b0-load-option').val()})
    $("#b0-load").attr('disabled', true);
    $('#b0-load-spinner').removeClass('d-none');
})

socketio.on('B0 session loaded',()=>{
    $("#output-space").text('B0 session data loaded from halbach.mat');
    $("#b0-load").attr('disabled', false);
    $('#b0-load-spinner').addClass('d-none');
})


socketio.on('B0 session loading failed',()=>{
    $("#output-space").text('B0 session data loading failed');
    $("#b0-load").attr('disabled', false);
    $('#b0-load-spinner').addClass('d-none');
})

$("#b0-mod").on("click",()=>{
    // TODO later; open modal to allow changing of session parameters
})
