for img in squares hanoi escher-square-limit lenet logo hilbert koch tensor hex-variation tree lattice; do
    # convert examples/output/${img}.png -gravity center -crop 200x200+0+0 +repage doc/imgs/${img}.png
    convert -define png:size=200x200 examples/output/${img}.png  -thumbnail 200x200^ -gravity center -extent 200x200  doc/imgs/${img}.png
done
