let socket = io();

$('#upload-button').on('click',()=>{
    let formData = new FormData($('#upload-form').get(0));
    $.ajax({
        type: 'POST',
        url : '/research/sequence',
        data: formData,
        success:function(data){
        },
        cache: false,
        contentType: false,
        processData: false
    })
})

socket.on("Seq file uploaded",()=>{
    $("#display-seq").prop('disabled',false);
})

$("#display-seq").on("click",()=>{
    let minTime = $('#min_time_field').val();
    let maxTime = $('#max_time_field').val();
    socket.emit("Display sequence",{'min': minTime, 'max': maxTime});
})

socket.on("Deliver seq plot",(payload)=>{
    Plotly.newPlot('psd-chart',JSON.parse(payload['graph']), {autosize: true});
})

socket.on("Message",(payload)=>{
    $("#message-region").text(payload['text']);
    $("#message-region").addClass(`text-${payload['type']}`);
})