let socket = io();

// Session updates
$('.session-variable').on('change',(event)=>{
    let myvalue;
    if (event.target.classList.contains('var-float')){
        myvalue = parseFloat(event.target.value);
    }
    if (event.target.classList.contains('var-int')){
        myvalue = parseInt(event.target.value);
    }
    if (event.target.classList.contains('var-string')){
        myvalue = event.target.value;
    }
    let myid = event.target.id; // TODO get the right string ......
    socket.emit("Update session variable rf", { 'id' : myid, 'value': myvalue })
})


$('#rf-display').on('click',()=>{
    console.log('Time to display');
    socket.emit('Display RF', gather_rf_options());
})

$("#rf-simulate").on('click',()=>{
    console.log('Time to simulate');
    socket.emit('Simulate RF',gather_rf_options())

    $("#rf-simulate").attr('disabled',true);
    $('#run-rf-sim-spinner').removeClass('d-none');


})

socket.on('Deliver RF pulse',(payload)=>{
    Plotly.newPlot('rf-pulse-chart',JSON.parse(payload['graph']), {autosize: true});
})

socket.on('Deliver RF profile',(payload)=>{
    Plotly.newPlot('rf-profile-chart',JSON.parse(payload['graph-profile']),{autosize: true});
    Plotly.newPlot('rf-evolution-chart',JSON.parse(payload['graph-evol']),{autosize: true});

    $("#rf-simulate").attr('disabled',false);
    $('#run-rf-sim-spinner').addClass('d-none');
})

function gather_rf_options(){
    return {
        'spin_bw': parseFloat($('#spin_bw').val()),
        'spin_num': parseInt($('#spin_num').val()),
        'pulse_type': $("#pulse_type").val(),
        'rf_shape': $('#rf_shape').val(),
        'rf_fa': parseFloat($("#rf_fa").val()),
        'rf_dur': parseFloat($("#rf_dur").val()),
        'rf_df': parseFloat($("#rf_df").val()),
        'rf_dphi': parseFloat($('#rf_dphi').val()),
        'rf_tbw': parseInt($("#rf_tbw").val()),
        'rf_thk': parseFloat($('#rf_thk').val()),
        'spin_t1': parseFloat($('#spin_t1').val()),
        'spin_t2': parseFloat($("#spin_t2").val())
    }

}