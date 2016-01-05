# Make plot showing effect of data augmentation on geoquery
pdf('fig-geo-augment.pdf')
op = par(family='serif')
x = c(100, 200, 300, 400, 500, 600)
y_orig = c(42.9, 60.4, 74.3, 78.5, 82.1, 85.4)
y_aug = c(50.7, 73.9, 80.0, 81.1, 87.1, 87.9)
plot(x, y_orig, ylim=c(35, 90), col='red', 
     xlab='Number of original training examples', ylab='Accuracy (%)',
     cex=1.5, cex.axis=1.5, cex.lab=1.5)
lines(x, y_orig, col='red')
points(x, y_aug, col='blue', cex=1.5)
lines(x, y_aug, col='blue')

legend('bottomright', c('No augmented data', 'With augmented data'),
       col= c('red', 'blue'), lty=1, cex=1.5)
dev.off()
par(op)
