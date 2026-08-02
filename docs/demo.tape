Output assets/demo.gif
Set FontSize 14
Set Width 980
Set Height 520
Set Padding 12
Set Theme "Dracula"

# install (editable so the bundled sample + repo samples/ both resolve)
Type "pip install -e ."
Enter
Sleep 3s

# score the predictor on the bundled Kimi K3 Q4 sample trace
Type "xpref eval"
Enter
Sleep 3s

# replay the decode loop: per-token recall + projected 4→12 t/s curve
Type "truncate -s 256k /tmp/ckpt.gguf"
Enter
Type "xpref attach --replay --checkpoint /tmp/ckpt.gguf"
Enter
Sleep 5s
