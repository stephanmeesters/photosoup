struct PsInput {
    [[vk::location(0)]] float4 color : COLOR0;
};

[[vk::location(0)]] float4 main(PsInput input) : SV_Target {
    return input.color;
}
