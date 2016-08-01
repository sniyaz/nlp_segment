#!/usr/bin/perl -w
#
# Usage:
# evaluation.perl [-trace] -desired <goldstdfile> -suggested <yoursegmentsfile>
#
# Arguments:
# -desired <file>    file containing the correct segmentation,
#                    e.g., goldstdsample.fin
# -suggested <file>  file containing the words segmented by your algorithm,
#                    each word occupies a line of its own and morpheme breaks
#                    are marked with a space, e.g.,
#                    open mind ed ness
# -trace             optional argument; if it is used, info about each word
#                    is output during processing
#
# Mathias Creutz, Aug 30, 2005
# Pascal Challenge 2005: Unsupervised segmentation of words into morphemes

use strict;
use Encode;

my($me, $arg);

$| = 1;

# Read command line arguments

($me = $0) =~ s,^.*/,,;

my($trace) = 0;
my($desiredfile) = '';
my($suggestedfile) = '';

while ($arg = shift @ARGV) {
    if ($arg eq "-desired") {
	$desiredfile = shift @ARGV;
    }
    elsif ($arg eq "-suggested") {
	$suggestedfile = shift @ARGV;
    }
    elsif ($arg eq "-trace") {
	$trace = 1;
    }
    else {
	&usage();
    }
}

&usage unless ($desiredfile & $suggestedfile);


# Read desired, or gold standard, segmentation

my($utf8line, $line, %analyses);

my($lineno) = 0;
my($ngoldstdwords) = 0; # Number of words in the gold standard

open(DESFILE, $desiredfile) ||
    die "Error ($me): Unable to open file \"$desiredfile\" for reading.\n";
while ($utf8line = <DESFILE>) {
    $line = decode("utf8", $utf8line);
    chomp $line;
    $line =~ s/\r$//;
    $lineno++;
    # Store gold standard segmentation for each word in the gold standard file
    if ($line =~ m/^([^\t]+)\t([^\t]+)$/) {
	my($word) = $1;
	my($analysisstring) = $2;
	$analyses{$word} = $analysisstring;
	$ngoldstdwords++;
    }
    else {
	die "Error ($me): Invalid line (number $lineno) in file ".
	    "\"$desiredfile\".\n";
    }
}
close DESFILE;


# Read segmentations suggested by some learning algorithm

my($nsuggestedwords) = 0; # Number of words in the data set (for which a 
                          # segmentation is proposed)
my($nevaluatedwords) = 0; # Number of words that are both in the gold standard
                          # and in the provided data set

my($nallhits) = 0;	  # Total number of correct, inserted, and
my($nallinsertions) = 0;  # deleted morpheme boundaries
my($nalldeletions) = 0;

open(SUGFILE, $suggestedfile) ||
    die "Error ($me): Unable to open file \"$suggestedfile\" for reading.\n";
while ($utf8line = <SUGFILE>) {
    $line = decode("utf8", $utf8line);
    chomp $line;
    $line =~ s/\r$//;

    my(@sugmorphs) = split(/ /, $line);
    my($word) = join("", @sugmorphs);
    $nsuggestedwords++;

    if (defined $analyses{$word}) {
	# The word is present in the gold standard: Evaluate it!
	&evaluate($word, @sugmorphs);
	$nevaluatedwords++;
    }
}
close SUGFILE;

# Report results

print "$me, " . localtime() . "\n";
print "Evaluation of segmentation in file \"$suggestedfile\" against\n" .
"gold standard segmentation in file \"$desiredfile\":\n";
print "Number of words in gold standard: $ngoldstdwords (type count)\n";
print "Number of words in data set: $nsuggestedwords (type count)\n";
printf("Number of words evaluated: %d (%.2f%% of all words in data set)\n",
       $nevaluatedwords, 100*$nevaluatedwords/$nsuggestedwords);

print "Morpheme boundary detections statistics:\n";

my($precision, $recall, $fmeasure);
if ($nallhits + $nallinsertions > 0) {
    $precision = $nallhits/($nallhits + $nallinsertions);
}
else {
    $precision = 1;
}
if ($nallhits + $nalldeletions > 0) {
    $recall = $nallhits/($nallhits + $nalldeletions);
}
else {
    $recall = 1;
}
if ($nallhits + $nallinsertions + $nalldeletions > 0) {
    $fmeasure = 2*$nallhits/(2*$nallhits +  $nallinsertions + $nalldeletions);
}
else {
    $fmeasure = 1;
}

printf("F-measure:  %.2f%%\n", $fmeasure*100);
printf("Precision:  %.2f%%\n", $precision*100);
printf("Recall:     %.2f%%\n", $recall*100);

# End.


sub evaluate {
    my($word, @sugmorphs) = @_;
    
    # Compute accuracy relative to each alternative correct (desired)
    # segmentation
    my (@altanalyses) = split(/, /, $analyses{$word});
    my($analysis);
    # Store result from best matching alternative correct answer
    my($bestfmeasure) = 0;
    my($bestprecision) = 0;
    my($bestanalysis) = "";
    my($bestnhits) = 0;
    my($bestninsertions) = 0;
    my($bestndeletions) = 0;
    foreach $analysis (@altanalyses) {
	my($nhits) = 0;
	my($ninsertions) = 0;
	my($ndeletions) = 0;
	$analysis =~ s/^ *//;
	$analysis =~ s/ *$//;
	my(@deschars) = split(//, $analysis);
	my(@sugchars) = split(//, join(" ", @sugmorphs));
	my($deschar);
	my($sugchar) = shift @sugchars;
	foreach $deschar (@deschars) {
	    if ($deschar eq " ") { # Desired morpheme boundary
		if ($sugchar eq " ") { # Recognized morpheme boundary
		    $nhits++;	       # Correct = hit
		    $sugchar = shift @sugchars;
		}
		else {		       # Missing boundary = deletion
		    $ndeletions++;
		}
	    }
	    else {
		if ($sugchar eq " ") { # Inserted incorrect boundary
		    $ninsertions++;
		    $sugchar = shift @sugchars;
		    die "Error ($me): Mismatch in string comparison. " .
			"Char $deschar vs. $sugchar.\n"
			unless ($sugchar eq $deschar);
		    $sugchar = shift @sugchars;
		}
		else {
		    die "Error ($me): Mismatch in string comparison. " .
			"Char $deschar vs. $sugchar.\n"
			unless ($sugchar eq $deschar);
		    $sugchar = shift @sugchars;
		}
	    }
	}
	my($precision, $fmeasure);
	if ($nhits + $ninsertions > 0) {
	    $precision = $nhits/($nhits + $ninsertions);
	}
	else {
	    $precision = 1;
	}
	if ($nhits + $ninsertions + $ndeletions > 0) {
	    $fmeasure = 2*$nhits/(2*$nhits +  $ninsertions + $ndeletions);
	}
	else {
	    $fmeasure = 1;
	}
	# Store best result
	if (($fmeasure > $bestfmeasure) ||
	    (($fmeasure == $bestfmeasure) && ($precision >= $bestprecision))) {
	    $bestfmeasure = $fmeasure;
	    $bestprecision = $precision;
	    $bestanalysis = $analysis;
	    $bestnhits = $nhits;
	    $bestninsertions = $ninsertions;
	    $bestndeletions = $ndeletions;
	}
    }
    if ($trace) {
	printf("DES: %s, SUG: %s, #hits: %d, #ins: %d, #del: %d\n",
	       encode("utf8", $bestanalysis),
	       encode("utf8", join(" ", @sugmorphs)),
	       $bestnhits, $bestninsertions, $bestndeletions);
    }
    $nallhits += $bestnhits;
    $nallinsertions += $bestninsertions;
    $nalldeletions += $bestndeletions;
}

sub usage {
    die "Usage: $me [-trace] -desired <goldstdfile> -suggested <yoursegmentsfile>\n";
}
